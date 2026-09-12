//! Translation of `IMOD/3dmod/mv_light.cpp` and `mv_light.h`.
//!
//! Compatibility-profile OpenGL calls are represented by `LightGl`; the
//! native `three-dmod-gl` driver supplies such a current context.  The math
//! and material state are source-identical and not a substitute shading model.
#![allow(dead_code)]

use crate::imod::libimod::imodel::{Iobj, Iview};
use crate::imod::libimod::iobj::IMOD_OBJFLAG_FCOLOR;
use crate::imod::libimod::iview::VIEW_WORLD_LIGHT;

#[derive(Clone, Debug)]
pub struct LightState {
    pub position: [f32; 4],
    pub att: [f32; 3],
    pub dist: f32,
    pub current: i32,
    pub vec: i32,
    pub first: bool,
}
impl Default for LightState {
    fn default() -> Self {
        Self {
            position: [0.; 4],
            att: [0.; 3],
            dist: 0.,
            current: 0,
            vec: 0,
            first: true,
        }
    }
}
pub trait LightGl {
    fn light_attenuation(&mut self, light: i32, constant: f32, linear: f32, quadratic: f32);
    fn light_ambient(&mut self, light: i32, ambient: [f32; 4]);
    fn light_position(&mut self, light: i32, position: [f32; 4], local_viewer: bool);
    fn light_model(&mut self, ambient: [f32; 4], local_viewer: bool);
    fn material(
        &mut self,
        ambient: [f32; 4],
        diffuse: [f32; 4],
        specular: [f32; 4],
        shininess: f32,
    );
    fn lighting(&mut self, enabled: bool);
    fn push_load_identity(&mut self);
    fn model_scale(&mut self, x: f32, y: f32, z: f32);
    fn pop_matrix(&mut self);
    fn color_material(&mut self, enabled: bool);
}
/// `imodvSetLight`.
pub fn imodv_set_light(state: &mut LightState, view: &mut Iview, gl: &mut dyn LightGl) {
    light_moveby(state, view, 0, 0, gl);
}
/// `light_getparam`.
pub fn light_getparam(state: &LightState, param: i32, out_value: &mut f32) {
    match param {
        1..=3 => *out_value = state.position[(param - 1) as usize],
        4..=6 => *out_value = state.att[(param - 4) as usize],
        7 => *out_value = state.dist,
        _ => {}
    }
}
/// `light_setparam`.
pub fn light_setparam(state: &mut LightState, param: i32, value: f64, gl: &mut dyn LightGl) {
    match param {
        1..=3 => state.position[(param - 1) as usize] = value as f32,
        4..=6 => {
            state.att[(param - 4) as usize] = value as f32;
            gl.light_attenuation(0, state.att[0], state.att[1], state.att[2]);
            return;
        }
        7 => state.dist = value as f32,
        _ => return,
    };
    light_update(state, gl);
}
/// Static `light_update`.
pub fn light_update(state: &mut LightState, gl: &mut dyn LightGl) {
    gl.light_attenuation(state.current, state.att[0], state.att[1], state.att[2]);
    let position = if state.dist == 0. {
        state.position[3] = 0.;
        [state.position[0], state.position[1], state.position[2], 0.]
    } else {
        state.position[3] = 1.;
        [state.position[0], state.position[1], state.position[2], 1.]
    };
    gl.light_position(state.current, position, state.dist != 0.);
}
/// `light_init`.
pub fn light_init(state: &mut LightState, view: &Iview, lighting: &mut i32, gl: &mut dyn LightGl) {
    if state.first {
        state.dist = 0.;
        state.att = [1., 0., 0.];
        state.current = 0;
        state.position = [0., 0., 1., 0.];
        state.first = false;
    }
    *lighting = ((view.world & VIEW_WORLD_LIGHT) != 0) as i32;
    gl.light_ambient(0, [0.1, 0.1, 0.1, 1.]);
    gl.light_model([0.7, 0.7, 0.7, 1.], false);
}
/// `light_moveby`.
pub fn light_moveby(
    state: &mut LightState,
    view: &mut Iview,
    x: i32,
    y: i32,
    gl: &mut dyn LightGl,
) {
    let mut lx = (view.lightx * 10.) as i32 + 2 * x;
    let mut ly = (view.lighty * 10.) as i32 + 2 * y;
    light_move(state, &mut lx, &mut ly, gl);
    view.lightx = lx as f32 * 0.1;
    view.lighty = ly as f32 * 0.1;
}
/// `light_move`.
pub fn light_move(state: &mut LightState, x: &mut i32, y: &mut i32, gl: &mut dyn LightGl) {
    if state.dist > 0. {
        state.position[0] = *x as f32;
        state.position[1] = *y as f32;
        light_update(state, gl);
        return;
    }
    *x = (*x).clamp(-800, 800);
    *y = (*y).clamp(-800, 800);
    let xa = *x as f64 * 0.1;
    let ya = *y as f64 * 0.1;
    let mut xn = (10_f64.powf((0.011 * xa).abs() + 0.845099) - 7.) as f32;
    let mut yn = (10_f64.powf((0.011 * ya).abs() + 0.845098) - 7.) as f32;
    if xa < 0. {
        xn = -xn
    };
    if ya > 0. {
        yn = -yn
    };
    state.position[0] = xn;
    state.position[1] = yn;
    state.position[2] = 1.;
    light_update(state, gl);
}
/// `light_adjust`.
pub fn light_adjust(object: &Iobj, r: f32, g: f32, b: f32, trans: i32, gl: &mut dyn LightGl) {
    let alpha = 1. - trans as f32 / 100.;
    let ambient = object.ambient as f32 / 255.;
    let diffuse = object.diffuse as f32 / 255.;
    let spec = object.specular as f32 / 255.;
    gl.material(
        [r * ambient, g * ambient, b * ambient, alpha],
        [r * diffuse, g * diffuse, b * diffuse, alpha],
        [spec + r, spec + g, spec + b, alpha],
        0.,
    );
}
/// `light_on`.
pub fn light_on(
    state: &mut LightState,
    object: &Iobj,
    view: &Iview,
    model_zscale: f32,
    gl: &mut dyn LightGl,
) {
    let (mut r, mut g, mut b) = (object.red, object.green, object.blue);
    if object.flags & IMOD_OBJFLAG_FCOLOR != 0 {
        (r, g, b) = (
            object.fillred as f32 / 255.,
            object.fillgreen as f32 / 255.,
            object.fillblue as f32 / 255.,
        )
    };
    let alpha = 1. - object.trans as f32 / 100.;
    let ambient = object.ambient as f32 / 255.;
    let diffuse = object.diffuse as f32 / 255.;
    let spec = object.specular as f32 / 255.;
    let shine = (255 - object.shininess) as f32 / 50. + 1.;
    gl.push_load_identity();
    gl.material(
        [r * ambient, g * ambient, b * ambient, alpha],
        [r * diffuse, g * diffuse, b * diffuse, alpha],
        [spec + r, spec + g, spec + b, alpha],
        shine,
    );
    let _ = (view.scale.x, view.scale.y, view.scale.z * model_zscale);
    gl.lighting(true);
    gl.model_scale(view.scale.x, view.scale.y, view.scale.z * model_zscale);
    light_update(state, gl);
    gl.pop_matrix();
}
/// `light_off`.
pub fn light_off(gl: &mut dyn LightGl) {
    gl.lighting(false);
    gl.color_material(false);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Gl {
        calls: Vec<String>,
    }
    impl LightGl for Gl {
        fn light_attenuation(&mut self, light: i32, constant: f32, linear: f32, quadratic: f32) {
            self.calls.push(format!(
                "attenuation:{light}:{constant}:{linear}:{quadratic}"
            ));
        }
        fn light_ambient(&mut self, _: i32, _: [f32; 4]) {
            self.calls.push("light-ambient".into());
        }
        fn light_position(&mut self, _: i32, _: [f32; 4], _: bool) {}
        fn light_model(&mut self, _: [f32; 4], _: bool) {}
        fn material(&mut self, _: [f32; 4], _: [f32; 4], _: [f32; 4], _: f32) {}
        fn lighting(&mut self, enabled: bool) {
            self.calls.push(format!("lighting:{enabled}"));
        }
        fn push_load_identity(&mut self) {
            self.calls.push("push".into());
        }
        fn model_scale(&mut self, x: f32, y: f32, z: f32) {
            self.calls.push(format!("scale:{x}:{y}:{z}"));
        }
        fn pop_matrix(&mut self) {
            self.calls.push("pop".into());
        }
        fn color_material(&mut self, enabled: bool) {
            self.calls.push(format!("color-material:{enabled}"));
        }
    }
    #[test]
    fn move_clamps_and_sets_infinite_position() {
        let mut s = LightState::default();
        let mut g = Gl::default();
        let (mut x, mut y) = (1000, -1000);
        light_move(&mut s, &mut x, &mut y, &mut g);
        assert_eq!((x, y), (800, -800));
        assert_eq!(s.position[3], 0.);
    }
    #[test]
    fn compatibility_gl_sequence_keeps_source_light_operations() {
        let mut state = LightState::default();
        let mut gl = Gl::default();
        let view = Iview::default();
        let mut lighting = 0;
        light_init(&mut state, &view, &mut lighting, &mut gl);
        light_setparam(&mut state, 5, 0.25, &mut gl);
        light_off(&mut gl);
        assert_eq!(
            gl.calls,
            [
                "light-ambient",
                "attenuation:0:1:0.25:0",
                "lighting:false",
                "color-material:false"
            ]
        );
    }
}
