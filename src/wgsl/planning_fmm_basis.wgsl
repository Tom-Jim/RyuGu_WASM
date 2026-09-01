// GPU order-two FMM preparation. A dense depth-5 octree has 37449 nodes;
// zero-mass nodes are skipped. All 56 unit-density moment banks and target
// response columns are computed on GPU. Source topology is built by GPU atomics.
struct Params {
    source_start: u32, source_count: u32, level: u32, target_start: u32,
    target_count: u32, response_start: u32, _pad0: u32, _pad1: u32,
    radius: f32, gravity: f32, theta: f32, _pad2: f32,
};
struct LocalExpansion {
    center_half: vec4<f32>, field: vec4<f32>, jacobian_x: vec4<f32>,
    jacobian_y: vec4<f32>, jacobian_z: vec4<f32>, metadata: vec4<u32>,
};
struct Field {
    value: vec4<f32>, jx: vec4<f32>, jy: vec4<f32>, jz: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> upload: array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> links: array<vec2<u32>>;
// Zero is the empty-list sentinel; stored particle indices are one-based.
@group(0) @binding(4) var<storage, read_write> heads: array<atomic<u32>>;
// Split 10 high/low moments across two bindings to stay below 128 MiB each.
@group(0) @binding(5) var<storage, read_write> moments_a: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> moments_b: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> targets: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> responses: array<LocalExpansion>;

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    return vec2(s, (a - (s - v)) + (b - v));
}
fn add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    return two_sum(s.x, s.y + a.y + b.y);
}
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = a.x * b.x;
    return two_sum(p, fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x);
}
fn reciprocal(a: vec2<f32>) -> vec2<f32> {
    let r = 1.0 / a.x;
    return add(vec2(r, 0.0), mul(add(vec2(1.0, 0.0), -mul(a, vec2(r, 0.0))), vec2(r, 0.0)));
}
fn level_offset(level: u32) -> u32 { return ((1u << (3u * level)) - 1u) / 7u; }
fn grid_index(c: vec3<u32>, n: u32) -> u32 { return (c.z * n + c.y) * n + c.x; }
fn grid_coord(i: u32, n: u32) -> vec3<u32> { return vec3(i % n, (i / n) % n, i / (n*n)); }
fn moment_index(voxel: u32, node: u32) -> u32 { return voxel * 37449u + node; }
fn read_moment(i: u32, component: u32) -> vec2<f32> {
    if component < 4u {
        let at = i * 8u + component * 2u;
        return vec2(bitcast<f32>(atomicLoad(&moments_a[at])), bitcast<f32>(atomicLoad(&moments_a[at+1u])));
    }
    let at = i * 12u + (component-4u)*2u;
    return vec2(bitcast<f32>(atomicLoad(&moments_b[at])), bitcast<f32>(atomicLoad(&moments_b[at+1u])));
}
fn store_moment(i: u32, component: u32, value: vec2<f32>) {
    if component < 4u {
        let at = i * 8u + component * 2u;
        atomicStore(&moments_a[at], bitcast<u32>(value.x));
        atomicStore(&moments_a[at+1u], bitcast<u32>(value.y));
    } else {
        let at = i * 12u + (component-4u)*2u;
        atomicStore(&moments_b[at], bitcast<u32>(value.x));
        atomicStore(&moments_b[at+1u], bitcast<u32>(value.y));
    }
}
fn add_a(i: u32, value: f32) -> f32 {
    var old = atomicLoad(&moments_a[i]);
    var residual = 0.0;
    loop {
        let sum = two_sum(bitcast<f32>(old), value);
        let result = atomicCompareExchangeWeak(&moments_a[i], old, bitcast<u32>(sum.x));
        if result.exchanged {
            residual = sum.y;
            break;
        }
        old = result.old_value;
    }
    // Naga requires a value return at the function boundary, after the loop.
    return residual;
}
fn add_b(i: u32, value: f32) -> f32 {
    var old = atomicLoad(&moments_b[i]);
    var residual = 0.0;
    loop {
        let sum = two_sum(bitcast<f32>(old), value);
        let result = atomicCompareExchangeWeak(&moments_b[i], old, bitcast<u32>(sum.x));
        if result.exchanged {
            residual = sum.y;
            break;
        }
        old = result.old_value;
    }
    return residual;
}
fn accumulate_moment(i: u32, component: u32, value: vec2<f32>) {
    if component < 4u {
        let at = i * 8u + component * 2u;
        let residual = add_a(at, value.x);
        _ = add_a(at+1u, residual + value.y);
    } else {
        let at = i * 12u + (component-4u)*2u;
        let residual = add_b(at, value.x);
        _ = add_b(at+1u, residual + value.y);
    }
}
@compute @workgroup_size(128)
fn p2m(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.source_count { return; }
    let at = id.x * 5u;
    let particle = vec4(bitcast<f32>(upload[at]), bitcast<f32>(upload[at+1u]), bitcast<f32>(upload[at+2u]), bitcast<f32>(upload[at+3u]));
    let voxel = upload[at+4u];
    let source = params.source_start + id.x;
    particles[source] = particle;
    let c = vec3<u32>(clamp(floor((particle.xyz / params.radius + vec3(1.0))*16.0), vec3(0.0), vec3(31.0)));
    let leaf = grid_index(c, 32u);
    let previous = atomicExchange(&heads[leaf], source+1u);
    links[source] = vec2(previous, voxel);
    let node = moment_index(voxel, level_offset(5u)+leaf);
    // Raw body-frame moments retain low words through P2M and M2M.
    let mass = vec2(particle.w, 0.0);
    let x = vec2(particle.x, 0.0);
    let y = vec2(particle.y, 0.0);
    let z = vec2(particle.z, 0.0);
    var values = array<vec2<f32>, 10>(mass, mul(mass,x), mul(mass,y), mul(mass,z),
        mul(mul(mass,x),x), mul(mul(mass,x),y), mul(mul(mass,x),z),
        mul(mul(mass,y),y), mul(mul(mass,y),z), mul(mul(mass,z),z));
    for (var component=0u; component<10u; component+=1u) { accumulate_moment(node, component, values[component]); }
}
@compute @workgroup_size(128)
fn m2m(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = 1u << params.level;
    let count = n*n*n;
    if id.x >= 56u * count { return; }
    let voxel = id.x / count;
    let c = grid_coord(id.x % count, n);
    let parent = moment_index(voxel, level_offset(params.level) + id.x % count);
    var sums: array<vec2<f32>, 10>;
    for (var component=0u; component<10u; component+=1u) { sums[component]=vec2(0.0); }
    for (var child=0u; child<8u; child+=1u) {
        let d = vec3(child&1u, (child>>1u)&1u, (child>>2u)&1u);
        let node = moment_index(voxel, level_offset(params.level+1u)+grid_index(c*2u+d, n*2u));
        for (var component=0u; component<10u; component+=1u) { sums[component]=add(sums[component],read_moment(node,component)); }
    }
    for (var component=0u; component<10u; component+=1u) { store_moment(parent,component,sums[component]); }
}
fn zero_field() -> Field { return Field(vec4(0.0),vec4(0.0),vec4(0.0),vec4(0.0)); }
fn sum_field(a: Field, b: Field) -> Field { return Field(a.value+b.value,a.jx+b.jx,a.jy+b.jy,a.jz+b.jz); }
fn difference(a: Field, b: Field) -> Field { return Field(a.value-b.value,a.jx-b.jx,a.jy-b.jy,a.jz-b.jz); }
fn direct(particle: vec4<f32>, observer: vec3<f32>) -> Field {
    let d = particle.xyz - observer;
    let r2 = max(dot(d,d), 1e-8);
    let inv = inverseSqrt(r2);
    let inv3 = inv/r2;
    let mass = params.gravity * particle.w;
    let diagonal = -mass*inv3;
    let outer = 3.0*mass*inv3/r2;
    return Field(mass*vec4(d*inv3,inv),
        vec4(vec3(diagonal,0.0,0.0)+outer*d*d.x,0.0),
        vec4(vec3(0.0,diagonal,0.0)+outer*d*d.y,0.0),
        vec4(vec3(0.0,0.0,diagonal)+outer*d*d.z,0.0));
}
fn multipole(node: u32, observer: vec3<f32>) -> Field {
    let m = read_moment(node,0u);
    let inv_mass = reciprocal(m);
    let first = array<vec2<f32>,3>(read_moment(node,1u),read_moment(node,2u),read_moment(node,3u));
    let com = array<vec2<f32>,3>(mul(first[0],inv_mass),mul(first[1],inv_mass),mul(first[2],inv_mass));
    let xx = add(read_moment(node,4u),-mul(first[0],com[0]));
    let xy = add(read_moment(node,5u),-mul(first[0],com[1]));
    let xz = add(read_moment(node,6u),-mul(first[0],com[2]));
    let yy = add(read_moment(node,7u),-mul(first[1],com[1]));
    let yz = add(read_moment(node,8u),-mul(first[1],com[2]));
    let zz = add(read_moment(node,9u),-mul(first[2],com[2]));
    let trace = add(add(xx,yy),zz);
    let qx = vec3(add(xx*3.0,-trace).x, 3.0*xy.x, 3.0*xz.x)*params.gravity;
    let qy = vec3(3.0*xy.x, add(yy*3.0,-trace).x, 3.0*yz.x)*params.gravity;
    let qz = vec3(3.0*xz.x, 3.0*yz.x, add(zz*3.0,-trace).x)*params.gravity;
    let d = vec3(com[0].x,com[1].x,com[2].x)-observer;
    let r2 = max(dot(d,d),1e-16);
    let inv = inverseSqrt(r2);
    let inv3=inv/r2; let inv5=inv3/r2; let inv7=inv5/r2; let inv9=inv7/r2;
    let qd = qx*d.x+qy*d.y+qz*d.z;
    let scalar = dot(d,qd);
    let mass = m.x*params.gravity;
    let diagonal = -mass*inv3-2.5*scalar*inv7;
    let outer = 3.0*mass*inv5+17.5*scalar*inv9;
    let mixed = -5.0*inv7;
    return Field(vec4(mass*d*inv3-qd*inv5+2.5*scalar*d*inv7, mass*inv+0.5*scalar*inv5),
        vec4(vec3(diagonal,0.0,0.0)+d*(outer*d.x)+qx*inv5+(qd*d.x+d*qd.x)*mixed,0.0),
        vec4(vec3(0.0,diagonal,0.0)+d*(outer*d.y)+qy*inv5+(qd*d.y+d*qd.y)*mixed,0.0),
        vec4(vec3(0.0,0.0,diagonal)+d*(outer*d.z)+qz*inv5+(qd*d.z+d*qd.z)*mixed,0.0));
}
fn accepted(level: u32, cell: u32, observer: vec3<f32>) -> bool {
    let n = 1u << level;
    let half_width = params.radius/f32(n);
    let center = -vec3(params.radius)+(vec3<f32>(grid_coord(cell,n))+vec3(0.5))*2.0*half_width;
    return 1.7320508075688772*half_width/ max(length(center-observer),1e-12) < params.theta;
}
var<workgroup> sums: array<Field,64>;
@compute @workgroup_size(64)
fn response_basis(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let observer_index = params.target_start+group.x;
    let voxel = group.y;
    let observer = targets[observer_index].xyz;
    var sum = zero_field();
    var correction = zero_field();
    // Each lane owns one level-2 subtree. An accepted level-1 ancestor is
    // emitted exactly once, by its even/even/even child lane.
    let c = grid_coord(lane,4u);
    let parent_cell = grid_index(c/2u,2u);
    let parent_node = moment_index(voxel,level_offset(1u)+parent_cell);
    let parent_mass = read_moment(parent_node,0u).x;
    if parent_mass > 0.0 {
        if accepted(1u,parent_cell,observer) {
            if all((c%2u)==vec3(0u)) { sum=multipole(parent_node,observer); }
        } else {
            // DFS depth at most four; 1+7*3=22 pending nodes, including siblings.
            var stack: array<vec2<u32>,32>;
            stack[0]=vec2(2u,lane);
            var count=1u;
            loop {
                if count==0u { break; }
                count-=1u;
                let current=stack[count];
                let node=moment_index(voxel,level_offset(current.x)+current.y);
                if read_moment(node,0u).x <= 0.0 { continue; }
                var contribution=zero_field();
                if accepted(current.x,current.y,observer) {
                    contribution=multipole(node,observer);
                } else if current.x==5u {
                    var source=atomicLoad(&heads[current.y]);
                    loop {
                        if source==0u { break; }
                        let link=links[source-1u];
                        if link.y==voxel {
                            let value=direct(particles[source-1u],observer);
                            let corrected=difference(value,correction);
                            let next=sum_field(sum,corrected);
                            correction=difference(difference(next,sum),corrected);
                            sum=next;
                        }
                        source=link.x;
                    }
                } else {
                    let n=1u<<current.x;
                    let base=grid_coord(current.y,n)*2u;
                    for (var child=0u;child<8u;child+=1u) {
                        let d=vec3(child&1u,(child>>1u)&1u,(child>>2u)&1u);
                        stack[count]=vec2(current.x+1u,grid_index(base+d,n*2u));
                        count+=1u;
                    }
                }
                let corrected=difference(contribution,correction);
                let next=sum_field(sum,corrected);
                correction=difference(difference(next,sum),corrected);
                sum=next;
            }
        }
    }
    sums[lane]=sum;
    workgroupBarrier();
    for (var stride=32u;stride>0u;stride/=2u) {
        if lane<stride { sums[lane]=sum_field(sums[lane],sums[lane+stride]); }
        workgroupBarrier();
    }
    if lane==0u {
        let index=(observer_index-params.response_start)*56u+voxel;
        responses[index]=LocalExpansion(vec4(observer,0.0),sums[0].value,sums[0].jx,sums[0].jy,sums[0].jz,vec4(0u));
    }
}
