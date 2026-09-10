//! Translation of `IMOD/libimod/iobj.c`.
#![allow(dead_code, unused_variables)]

use super::imodel::{Iclip_planes, Iobj};
use super::istore::istore_checksum;

/// Original: `imodObjectDefault` (`iobj.c:55`).
pub fn imod_object_default(object: &mut Iobj) {
    object.cont.clear();
    object.mesh.clear();
    object.store.clear();
    object.name.clear();
    object.extra = [0; 16];
    object.flags = (1 << 27) | (1 << 28);
    object.axis = 0;
    object.drawmode = 1;
    object.red = 0.5;
    object.green = 0.5;
    object.blue = 0.5;
    object.pdrawsize = 0;
    object.symbol = 1;
    object.symsize = 3;
    object.linewidth2 = 1;
    object.linewidth = 1;
    object.linesty = 0;
    object.symflags = 0;
    object.sympad = 0;
    object.trans = 0;
    object.surfsize = 0;
    object.clips = Iclip_planes::default();
    object.ambient = 102;
    object.diffuse = 255;
    object.specular = 127;
    object.shininess = 4;
    object.fillred = 0;
    object.fillgreen = 0;
    object.fillblue = 0;
    object.quality = 0;
    object.mat2 = 0;
    object.valblack = 0;
    object.valwhite = 255;
    object.matflags2 = 0;
    object.mesh_thickness = 0;
}

/// Original: `imodObjectsNew` (`iobj.c:37`).
pub fn imod_objects_new(size: i32) -> Option<Vec<Iobj>> {
    if size < 0 {
        return None;
    }
    let mut objects = Vec::with_capacity(size as usize);
    for _ in 0..size {
        let mut object = Iobj::default();
        imod_object_default(&mut object);
        objects.push(object);
    }
    Some(objects)
}

/// Original: `imodObjectNew` (`iobj.c:24`).
pub fn imod_object_new() -> Option<Iobj> {
    imod_objects_new(1).and_then(|mut objects| objects.pop())
}

/// Original: `imodObjectDelete` (`iobj.c:109`).
pub fn imod_object_delete(object: &mut Iobj) -> i32 {
    object.cont.clear();
    object.mesh.clear();
    object.store.clear();
    0
}

/// Original: `imodObjectsDelete` (`iobj.c:119`).
pub fn imod_objects_delete(objects: &mut Vec<Iobj>) -> i32 {
    if objects.is_empty() {
        return -1;
    }
    for object in objects.iter_mut() {
        imod_object_delete(object);
    }
    objects.clear();
    0
}

/// Original: `imodObjectCopy` (`iobj.c:202`).
pub fn imod_object_copy(from: &Iobj, to: &mut Iobj) -> i32 {
    *to = from.clone();
    0
}

/// Original: `imodObjectCopyClear` (`iobj.c:216`).
pub fn imod_object_copy_clear(from: &Iobj, to: &mut Iobj) -> i32 {
    *to = from.clone();
    to.cont.clear();
    to.mesh.clear();
    to.store.clear();
    0
}

/// Original: `imodObjectDup` (`iobj.c:234`).
pub fn imod_object_dup(object: &Iobj) -> Option<Iobj> {
    Some(object.clone())
}

/// Original: `imodObjectChecksum` (`iobj.c:145`).
pub fn imod_object_checksum(object: &Iobj, object_number: i32) -> f64 {
    let mut object_sum = object_number as f64;
    let mut point_sum = 0.;
    object_sum += (object.red + object.green + object.blue) as f64;
    object_sum += object.flags as f64;
    object_sum += object.pdrawsize as f64;
    object_sum += object.symbol as f64;
    object_sum += object.symsize as f64;
    object_sum += object.linewidth2 as f64;
    object_sum += object.linewidth as f64;
    object_sum += object.symflags as f64;
    object_sum += object.trans as f64;
    object_sum += object.cont.len() as f64;
    object_sum += (object.ambient + object.diffuse + object.specular + object.shininess) as f64;
    object_sum +=
        (object.clips.count + object.clips.flags + object.clips.trans + object.clips.plane) as f64;
    object_sum += object.mat2 as f64;
    for plane in 0..object.clips.count.min(7) as usize {
        let normal = object.clips.normal[plane];
        let point = object.clips.point[plane];
        object_sum += (normal.x + normal.y + normal.z + point.x + point.y + point.z) as f64;
    }
    object_sum += (object.extra[0] + object.extra[1]) as f64;
    object_sum += (object.fillred + object.fillgreen + object.fillblue + object.quality) as f64;
    object_sum +=
        (object.valblack + object.valwhite + object.matflags2 + object.mesh_thickness) as f64;
    object_sum += istore_checksum(&object.store);
    for (contour_number, contour) in object.cont.iter().enumerate() {
        point_sum += contour.surf as f64;
        point_sum += contour.pts.len() as f64;
        if contour_number != 0 {
            point_sum += ((contour.pts.len() as i32
                - object.cont[contour_number - 1].pts.len() as i32)
                % 13) as f64;
        }
        point_sum += (contour.flags & !(1 << 4) & !(1 << 31)) as f64;
        point_sum += contour.time as f64;
        for (point_number, point) in contour.pts.iter().enumerate() {
            point_sum += (point.x * (point_number % 7 + 1) as f32
                + point.y * (point_number % 5 + 1) as f32
                + point.z) as f64;
        }
        point_sum += istore_checksum(&contour.store);
        point_sum += contour.sizes.iter().map(|size| *size as f64).sum::<f64>();
    }
    object_sum + point_sum
}
