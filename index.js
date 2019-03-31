module.exports = (function who_loves_voxels() {

  // Math
  
  function quat_from_axis_angle([ax, ay, az], angle) {
    var qw = Math.cos(angle / 2);
    var qx = ax * Math.sin(angle / 2);
    var qy = ay * Math.sin(angle / 2);
    var qz = az * Math.sin(angle / 2);
    return [qw, qx, qy, qz];
  }

  function quat_mul([aw, ax, ay, az], [bw, bx, by, bz]) { 
      var x =  ax * bw + ay * bz - az * by + aw * bx;
      var y = -ax * bz + ay * bw + az * bx + aw * by;
      var z =  ax * by - ay * bx + az * bw + aw * bz;
      var w = -ax * bx - ay * by - az * bz + aw * bw;
      return [w, x, y, z];
  };

  function quat_conjugate([w, x, y, z]) {
    return [w, -x, -y, -z];
  }

  function vec_rotate([x, y, z], q) {
    return quat_mul(quat_mul(q, [0, x, y, z]), quat_conjugate(q)).slice(1);
  }

  function vec_add([ax, ay, az], [bx, by, bz]) {
    return [ax + bx, ay + by, az + bz];
  }

  function vec_sub([ax, ay, az], [bx, by, bz]) {
    return [ax - bx, ay - by, az - bz];
  }

  function vec_scale([ax, ay, az], s) {
    return [ax * s, ay * s, az * s];
  }

  function vec_len([ax, ay, az]) {
    return Math.sqrt(ax * ax + ay * ay + az * az);
  }

  function vec_dist([ax, ay, az], [bx, by, bz]) {
    return Math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az));
  }

  function cam_tox(cam) {
    switch (cam.typ) {
      case PERSP_CAM: return vec_rotate([1, 0, 0], cam.rot);
      case TIBIA_CAM: return cam.dir;
    }
  }

  function cam_toy(cam) {
    switch (cam.typ) {
      case PERSP_CAM: return vec_rotate([0, 1, 0], cam.rot);
      case TIBIA_CAM: return cam.dir;
    }
  }

  function cam_toz(cam) {
    switch (cam.typ) {
      case PERSP_CAM: return vec_rotate([0, 0, 1], cam.rot);
      case TIBIA_CAM: return cam.dir;
    }
  }

  var quat_rot_x = angle => quat_from_axis_angle([1, 0, 0], angle);
  var quat_rot_y = angle => quat_from_axis_angle([0, 1, 0], angle);
  var quat_rot_z = angle => quat_from_axis_angle([0, 0, 1], angle);

  function sprite({loc, siz, pos, col, vox}) {
    if (typeof vox === "function") {
      var data = new Uint8Array(siz[0] * siz[1] * siz[2] * 4);
      for (var z = 0; z < siz[2]; ++z) {
        for (var y = 0; y < siz[1]; ++y) {
          for (var x = 0; x < siz[0]; ++x) {
            var rgba = vox([x, y, z]);
            var idx = (x + y * siz[0] + z * siz[0] * siz[1]) * 4;
            data[idx + 0] = (rgba & 0xFF000000) >>> 24;
            data[idx + 1] = (rgba & 0xFF0000) >>> 16;
            data[idx + 2] = (rgba & 0xFF00) >>> 8;
            data[idx + 3] = (rgba & 0xFF);
          }
        }
      };
    } else {
      var data = null;
    }
    return {loc, siz, pos, vox, col, data};
  }

  const PERSP_CAM = 0;
  const TIBIA_CAM = 1;

  // typ: the type of the camera
  // case PERSP_CAM:
  //   pos: the camera position
  //   dis: the distance to the screen
  //   rot: quaternion representing a rotation
  // case TIBIA_CAM:
  //   pos: the camera position
  //   dis: half of the screen width
  //   rot: direction of the camera (NOT its rotation)
  function cam(typ, pos, dis, r_d) {
    if (typ === PERSP_CAM) {
      var dis = dis === undefined ? 2.0 : dis;
      var dir = [-1/Math.sqrt(3), -1/Math.sqrt(3), -1];
      var rot = r_d === undefined ? quat_rot_z(0.0) : r_d;
    } else {
      var dis = dis === undefined ? 256.0 : dis;
      var dir = r_d === undefined ? [-1/Math.sqrt(3), -1/Math.sqrt(3), -1] : r_d;
      var rot = quat_rot_z(0.0);
    }
    return {typ, pos, dis, rot, dir};
  }

  // ============== Creating a canvas ====================

  function install({canvas, voxel_size, debug}) {
    gl = canvas.getContext('webgl2');

    var vertices = [-1,1,0,-1,-1,0,1,-1,0,-1,1,0,1,1,0,1,-1,0,];
    var indices = [0,1,2,3,4,5];

    var vertex_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertex_buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    var index_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index_buffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    // Vertex shader source code
    var vert_code = `#version 300 es
      in vec3 coordinates;
      out vec3 scr_pos;
      void main(void) {
        scr_pos = coordinates;
        gl_Position = vec4(coordinates, 1.0);
      }
    `;
      
    var vertShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertShader, vert_code);
    gl.compileShader(vertShader);

    var frag_code = `#version 300 es
      precision highp float;
      precision lowp sampler3D;

      in vec3 scr_pos;
      out vec4 outColor;

      uniform vec3 voxel_size;
      uniform sampler3D voxel_data;

      uniform int  sprite_len;
      uniform vec3 sprite_loc[64];
      uniform vec3 sprite_pos[64];
      uniform vec3 sprite_siz[64];

      uniform int   light_len;
      uniform vec3  light_pos[32];
      uniform float light_pow[32];

      uniform int cam_typ;
      uniform vec3 cam_pos;
      uniform vec3 cam_tox;
      uniform vec3 cam_toy;
      uniform vec3 cam_toz;
      uniform float cam_dis;
      uniform float time;

      const float inf = 3.402823466e+38; // TODO: how to improve this constant?
      const float eps = 0.001;

      struct Sprite {
        vec3 loc; 
        vec3 siz;
        vec3 pos;
      };

      struct Hit {
        float dist;
        int idx;
        Sprite spr;
      };

      struct March {
        float run_dist;
        vec3 hit_col;
        vec3 abs_col;
      };

      float ray_box_intersect(vec3 ray_pos, vec3 ray_dir, vec3 box_pos, vec3 box_siz) {
        vec3 box_min = box_pos - box_siz * 0.5;
        vec3 box_max = box_pos + box_siz * 0.5;
        float t1 = (box_min.x - ray_pos.x) / ray_dir.x;
        float t2 = (box_max.x - ray_pos.x) / ray_dir.x;
        float t3 = (box_min.y - ray_pos.y) / ray_dir.y;
        float t4 = (box_max.y - ray_pos.y) / ray_dir.y;
        float t5 = (box_min.z - ray_pos.z) / ray_dir.z;
        float t6 = (box_max.z - ray_pos.z) / ray_dir.z;
        float t7 = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
        float t8 = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
        float t9 = (t8 < 0.0 || t7 > t8) ? inf : t7;
        return t9;
      }

      // Returns the distance between a ray and its intersection in a sphere
      // Returns the distance even if ray origin is inside sphere
      // Returns -1.0 if collision point is behind
      // Returns -1.0 if there is no collision point
      float ray_sphere_intersect
        ( vec3 ray_pos
        , vec3 ray_dir
        , vec3 sphere_pos
        , float sphere_rad) {
        float a = dot(ray_dir, ray_dir);
        vec3 k = ray_pos - sphere_pos;
        float b = 2.0 * dot(ray_dir, k);
        float c = dot(k, k) - (sphere_rad * sphere_rad);
        float d = b*b - 4.0 * a * c; // negative if non-colliding
        float p = (-b - sqrt(d)) / (2.0 * a); // 1st point, negative if behind
        float q = (-b + sqrt(d)) / (2.0 * a); // 2nd point, negative if behind
        return d < 0.0 ? -1.0 : p < 0.0 ? (q < 0.0 ? -1.0 : q) : p;
      }

      float surface_distance(vec3 pos) {
        vec3  a_pos = vec3( 0.0, 0.0, 10.0);
        float a_rad = 0.6;
        vec3 v = a_pos - pos;
        return sqrt(v.x * v.x + v.y * v.y + v.z * v.z) - a_rad;
      }

      float sdf_union(float a, float b) {
        return min(a, b);
      }

      float sdf_intersection(float a, float b) {
        return max(a, b);
      }

      float sdf_smooth(float k, float a, float b) {
        float res = exp(-k*a) + exp(-k*b);
        return -log(max(0.0001,res)) / k;
      }

      float sdf_sphere(vec3 p, vec3 c, float s) {
        return length(p - c) - s;
      }

      float sdf_box(vec3 p, vec3 b) {
        vec3 d = abs(p) - b;
        return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
      }

      float sdf_y_plane(vec3 p, float y) {
        return p.y - y;
      }

      bool inside(vec3 pos, vec3 box_pos, vec3 box_siz) {
        vec3 d = box_siz * 0.5 - abs(pos - box_pos);
        return d.x >= 0.0 && d.y >= 0.0 && d.z >= 0.0;
      }

      vec4 get_vox(Sprite spr, vec3 pos) {
        // If not a sprite, return a flat color
        if (spr.loc.x < 0.0) {
          return vec4(-spr.loc / 255.0, 1.0);
        // Else, find its color on the voxel_data
        } else {
          vec3 d = spr.siz * 0.5 - abs(pos - spr.pos);
          if (d.x >= 0.0 && d.y >= 0.0 && d.z >= 0.0) {
            //return texture(voxel_data, vec3(0.5,0.5,0.5));
            return texture(voxel_data, (spr.loc + vec3(0.5,0.5,0.5) * spr.siz + (pos - spr.pos)) / voxel_size);
          } else {
            return vec4(0.0);
          }
        }
      }

      Hit next_hit(vec3 ray_pos, vec3 ray_dir) {
        int idx = -1;
        float dist = inf;
        for (int i = 0; i < sprite_len; ++i) {
          float sprite_dist = ray_box_intersect(ray_pos, ray_dir, sprite_pos[i], sprite_siz[i]);
          idx = sprite_dist < dist ? i : idx;
          dist = min(sprite_dist, dist);
        }
        return Hit(dist, idx, Sprite(sprite_loc[idx], sprite_siz[idx], sprite_pos[idx]));
      }

      March march(vec3 ray_pos, vec3 ray_dir, float max_dist) {
        vec3 ini_pos    = ray_pos;
        vec3 abs_col    = vec3(0.0);
        vec3 hit_col    = vec3(0.0);
        float ray_dist  = 0.0;
        float step      = 0.5;
        float max_steps = max_dist / step;

        // Finds the first voxbox on its way
        Hit hit = next_hit(ray_pos, ray_dir);
        if (!inside(ray_pos, hit.spr.pos, hit.spr.siz)) {
          ray_pos = ray_pos + ray_dir * (hit.dist + eps);
          ray_dist = ray_dist + hit.dist + eps; 
        }

        // Performs the march
        for (float k = 0.0; k < max_steps; ++k) {
          
          // If nothing more on its way, stop marching 
          if (hit.idx == -1 || ray_dist >= max_dist) {
            ray_dist = max_dist;
            break;
          }

          // If inside the focused voxbox, samples it
          if (inside(ray_pos, hit.spr.pos, hit.spr.siz)) {

            // Increments ray position and distance
            ray_pos += ray_dir * step;
            ray_dist += step;

            // Gets the voxel color
            vec4 vox = get_vox(hit.spr, ray_pos);

            // If it is solid, stop the march
            if (vox.w == 1.0) {
              hit_col = vec3(vox) - abs_col;
              abs_col = vec3(1.0, 1.0, 1.0);
              break;
            }

            // If it has transparent color, increments the absorbed color 
            if (vox.w > 0.0) {
              abs_col += (vec3(1.0) - vec3(vox)) * vox.w * step;
            }

          // If outside the focused voxbox, find next voxbox on its way
          } else {
            hit = next_hit(ray_pos, ray_dir);
            ray_pos = ray_pos + ray_dir * (hit.dist + eps);
            ray_dist = ray_dist + hit.dist + eps;
          }
        }
        return March(ray_dist, hit_col, abs_col);
      }

      void main(void) {
        vec3 pix_col = vec3(1.0);
        float max_dist = 1024.0;
        March marched;
        vec3 hit_pos;
        vec3 ray_pos;
        vec3 ray_dir;
        vec3 lig_pos;
        float lig_dist;

        // Sets initial ray position and direction
        if (cam_typ == ${PERSP_CAM}) {
          ray_pos = cam_pos;
          ray_dir = normalize(cam_tox * scr_pos.x + cam_toy * scr_pos.y + cam_toz * cam_dis);
        } else {
          ray_pos = vec3(cam_pos.x + cam_dis * scr_pos.x, cam_pos.y - cam_dis * scr_pos.y, cam_pos.z) + vec3(0.5, 0.5, 0.5);
          ray_dir = normalize(cam_tox + cam_toy + cam_toz);
        }

        // Marchs towards screen
        marched = march(ray_pos, ray_dir, max_dist);
        hit_pos = ray_pos + ray_dir * marched.run_dist;
        pix_col = marched.hit_col * 0.1;

        // Marchs towards lights
        if (marched.run_dist < max_dist) {
          for (int i = 0; i < max(light_len, 1); ++i) {
            lig_pos = light_pos[i];
            ray_dir = normalize(lig_pos - hit_pos);
            ray_pos = hit_pos + ray_dir * 1.5;
            marched = march(ray_pos, ray_dir, distance(lig_pos, ray_pos));
            pix_col = pix_col + light_pow[i] * (vec3(1.0) - marched.abs_col) / pow(distance(hit_pos, lig_pos), 2.0);
          }
        }

        outColor = vec4(pix_col, 1.0);
      }`;
      
    var fragShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragShader, frag_code); 
    gl.compileShader(fragShader);

    // TODO: improve
    if (debug) {
      var compiled = gl.getShaderParameter(vertShader, gl.COMPILE_STATUS);
      console.log('Shader compiled successfully: ' + compiled);
      var compilationLog = gl.getShaderInfoLog(vertShader);
      console.log('Shader compiler log: ' + compilationLog);
      var compiled = gl.getShaderParameter(fragShader, gl.COMPILE_STATUS);
      console.log('Shader compiled successfully: ' + compiled);
      var compilationLog = gl.getShaderInfoLog(fragShader);
      console.log('Shader compiler log: ' + compilationLog);
    }

    var shader = gl.createProgram();
    gl.attachShader(shader, vertShader);
    gl.attachShader(shader, fragShader);
    gl.linkProgram(shader);
    gl.useProgram(shader);

    // ======= Input texture =======

    var texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, texture);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA, voxel_size[0], voxel_size[1], voxel_size[2], 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.uniform1i(gl.getUniformLocation(shader, "voxel_data"), texture);
    gl.uniform3fv(gl.getUniformLocation(shader, "voxel_size"), voxel_size);

    // ======= Associating shaders to buffer objects =======

    // Bind vertex buffer object
    gl.bindBuffer(gl.ARRAY_BUFFER, vertex_buffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index_buffer);

    // Get the attribute location
    var coord = gl.getAttribLocation(shader, "coordinates");
    gl.vertexAttribPointer(coord, 3, gl.FLOAT, false, 0, 0); 
    gl.enableVertexAttribArray(coord);

    var start = Date.now() / 1000;
    canvas.__who_loves_voxels = {gl, shader, indices, start};
  }

  function render({canvas, camera, sprites, lights, voxel_size = [256, 256, 256], debug}) {
    var cam = camera;

    if (!canvas.__who_loves_voxels) {
      install({canvas, voxel_size, debug});
    } 

    var {gl, shader, indices, start} = canvas.__who_loves_voxels;

    // Sprites
    var sprite_loc = [];
    var sprite_pos = [];
    var sprite_siz = [];
    for (var i = 0; i < sprites.length; ++i) {
      if (sprites[i].col !== undefined) {
        sprite_loc.push(-((sprites[i].col & 0xFF000000) >>> 24), -((sprites[i].col & 0xFF0000) >>> 16), -((sprites[i].col & 0xFF00) >>> 8));
      } else {
        sprite_loc.push(sprites[i].loc[0], sprites[i].loc[1], sprites[i].loc[2]);
      }
      sprite_pos.push(sprites[i].pos[0], sprites[i].pos[1], sprites[i].pos[2]);
      sprite_siz.push(sprites[i].siz[0], sprites[i].siz[1], sprites[i].siz[2]);
    }

    // Lights
    var light_pos = [];
    var light_pow = [];
    for (var i = 0; i < lights.length; ++i) {
      light_pos.push(lights[i].pos[0], lights[i].pos[1], lights[i].pos[2]);
      light_pow.push(lights[i].pow);
    }

    // Upload sprite data
    for (var i = 0; i < sprites.length; ++i) {
      if (sprites[i].data) {
        gl.texSubImage3D(gl.TEXTURE_3D, 0, sprites[i].loc[0], sprites[i].loc[1], sprites[i].loc[2], sprites[i].siz[0], sprites[i].siz[1], sprites[i].siz[2], gl.RGBA, gl.UNSIGNED_BYTE, sprites[i].data);
        sprites[i].data = null;
      }
    }
    gl.uniform1i(gl.getUniformLocation(shader, "sprite_len"), sprites.length);
    gl.uniform3fv(gl.getUniformLocation(shader, "sprite_loc"), sprite_loc);
    gl.uniform3fv(gl.getUniformLocation(shader, "sprite_pos"), sprite_pos);
    gl.uniform3fv(gl.getUniformLocation(shader, "sprite_siz"), sprite_siz);

    // Upload light data
    gl.uniform1i(gl.getUniformLocation(shader, "light_len"), lights.length);
    gl.uniform3fv(gl.getUniformLocation(shader, "light_pos"), light_pos);
    gl.uniform1fv(gl.getUniformLocation(shader, "light_pow"), light_pow);

    // Upload camera data
    gl.uniform1i(gl.getUniformLocation(shader, "cam_typ"), cam.typ);
    gl.uniform3fv(gl.getUniformLocation(shader, "cam_pos"), cam.pos);
    gl.uniform3fv(gl.getUniformLocation(shader, "cam_tox"), cam_tox(cam));
    gl.uniform3fv(gl.getUniformLocation(shader, "cam_toy"), cam_toy(cam));
    gl.uniform3fv(gl.getUniformLocation(shader, "cam_toz"), cam_toz(cam));
    gl.uniform1f(gl.getUniformLocation(shader, "cam_dis"), cam.dis);
    gl.uniform1f(gl.getUniformLocation(shader, "time"), Date.now() / 1000 - start);

    // Clear and render
    gl.clearColor(0.5, 0.5, 0.5, 1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT,0);
  };

  return {
    quat_from_axis_angle,
    quat_mul,
    quat_conjugate,
    quat_rot_x,
    quat_rot_y,
    quat_rot_z,
    vec_rotate,
    vec_add,
    vec_sub,
    vec_scale,
    vec_len,
    vec_dist,
    cam_tox,
    cam_toy,
    cam_toz,
    PERSP_CAM,
    TIBIA_CAM,
    cam,
    sprite,
    render
  };

})();
