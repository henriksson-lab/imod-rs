//! Translation of `IMOD/flib/subrs/model/scale_model.f90`.
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::libiimod::unit_header::{iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt};
use crate::imod::libimod::imodel_fwrap::{
    getimodhead, getimodscales, imodhasimageref, putimageref,
};

/// Original: `scale_model` (`scale_model.f90:15`).
///
/// Will undo or redo the scaling from image index coordinates to model
/// coordinates performed by imodel_fwrap, using the data in the model header
/// about the image file that the model was last loaded on.  Set `idir` to 0
/// to undo the scaling, for working in index coordinates; set `idir` to 1 to
/// redo the scaling before saving the model back out.
pub fn scale_model(idir: i32, fm: &mut FortModel) {
    let (mut x_im_scale, mut yimscale, mut z_im_scale) = (0f32, 0f32, 0f32);
    let (mut xy_scale, mut z_scale) = (0f32, 0f32);
    let (mut x_offset, mut y_offset, mut z_offset) = (0f32, 0f32, 0f32);
    let mut if_flip: i32 = 0;
    //
    let ierr = getimodhead(
        &mut xy_scale,
        &mut z_scale,
        &mut x_offset,
        &mut y_offset,
        &mut z_offset,
        &mut if_flip,
    );
    let ierr2 = getimodscales(&mut x_im_scale, &mut yimscale, &mut z_im_scale);
    if ierr != 0 || ierr2 != 0 {
        println!(" ERROR: SCALE_MODEL: getting model header");
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    if idir == 0 {
        //
        // shift the data to index coordinates before working with it
        //
        for i in 1..=fm.n_point {
            let p = &mut fm.p_coord[i as usize - 1];
            p[0] = (p[0] - x_offset) / x_im_scale;
            p[1] = (p[1] - y_offset) / yimscale;
            p[2] = (p[2] - z_offset) / z_im_scale;
        }
    } else {
        //
        // shift the data back for saving
        //
        for i in 1..=fm.n_point {
            let p = &mut fm.p_coord[i as usize - 1];
            p[0] = x_im_scale * p[0] + x_offset;
            p[1] = yimscale * p[1] + y_offset;
            p[2] = z_im_scale * p[2] + z_offset;
        }
    }
}

/// Original: `scaleModelToImage` (`scale_model.f90:62`).
///
/// Will undo or redo the scaling from image index coordinates to model
/// coordinates performed by imodel_fwrap, using the header information of the
/// image file open on unit `iunit`.  The resulting index coordinates will fit
/// those of this image rather than the image file the model was last loaded
/// on.  Set `idir` to 0 to undo the scaling, 1 to redo it before saving.
pub fn scale_model_to_image(iunit: i32, idir: i32, fm: &mut FortModel) {
    //
    //
    let origin = iiu_ret_origin(iunit);
    let delta = iiu_ret_delta(iunit);
    let tilt = iiu_ret_tilt(iunit);

    if idir == 0 {
        //
        // shift the data to index coordinates before working with it, but only if the
        // it got shifted in the first place
        if imodhasimageref() <= 0 {
            return;
        }
        //
        for i in 1..=fm.n_point {
            for j in 1..=3usize {
                fm.p_coord[i as usize - 1][j - 1] =
                    (fm.p_coord[i as usize - 1][j - 1] + origin[j - 1]) / delta[j - 1];
            }
        }
    } else {
        //
        // shift the data back for saving, but first set the image ref to match this image
        // (`i = putImageRef(...)`: the result is discarded when `i` is reused
        // as the loop index)
        let _ = putimageref(&delta, &origin, &tilt);
        for i in 1..=fm.n_point {
            for j in 1..=3usize {
                fm.p_coord[i as usize - 1][j - 1] =
                    fm.p_coord[i as usize - 1][j - 1] * delta[j - 1] - origin[j - 1];
            }
        }
    }
}
