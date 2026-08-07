use crate::image_processing::apply_orientation;
use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use rawler::{
    decoders::{Orientation, RawDecodeParams},
    imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
    imgop::matrix::{multiply, normalize, pseudo_inverse},
    imgop::xyz::SRGB_TO_XYZ_D65,
    rawimage::{RawImage, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

/// A camera channel counts as clipped once it reaches this fraction of the sensor
/// white level. Slightly below 1.0 because demosaicing averages neighbours, which
/// pulls genuinely saturated samples a little under the ceiling.
const CLIP_ENTER: f32 = 0.96;
const CLIP_FULL: f32 = 0.995;

/// Highest value the reconstruction may produce, in white-balanced camera space.
/// A fully clipped pixel is rebuilt at max(wb), so this only needs to clear that.
const RECONSTRUCTION_CEILING: f32 = 8.0;

/// Camera RGB -> linear sRGB, derived the same way rawler's calibration step does.
fn build_cam_to_srgb(raw_image: &RawImage) -> Option<[[f32; 3]; 3]> {
    let (_illuminant, color_matrix) = raw_image
        .color_matrix
        .iter()
        .find(|(illuminant, _)| matches!(illuminant, rawler::imgop::xyz::Illuminant::D65))
        .or_else(|| raw_image.color_matrix.iter().next())?;

    if color_matrix.len() % 3 != 0 || color_matrix.len() < 9 {
        return None;
    }

    let mut xyz2cam = [[0.0f32; 3]; 3];
    for (i, row) in xyz2cam.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = color_matrix[i * 3 + j];
        }
    }

    let rgb2cam = normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65));
    Some(pseudo_inverse(rgb2cam))
}

/// How clipped a single channel is, 0..1, ramped so the transition is not a hard edge.
#[inline]
fn clippedness(v: f32) -> f32 {
    let t = ((v - CLIP_ENTER) / (CLIP_FULL - CLIP_ENTER)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(
        raw_image.photometric,
        RawPhotometricInterpretation::LinearRaw
    )
}

#[inline]
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(3.0)
    }
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let metadata = decoder.raw_metadata(&source, &RawDecodeParams::default())?;
    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    let is_linear_format = is_linear_raw_format(&raw_image);

    let (apply_ungamma, apply_calibration) = match linear_mode.as_str() {
        "gamma" => (true, true),
        "skip_calib" => (false, false),
        "gamma_skip_calib" => (true, false),
        _ => (false, true),
    };

    // White level is left intact: rawler then normalises a clipped sample to exactly
    // 1.0, which is the only place where "this channel hit the sensor ceiling" is a
    // clean, axis-aligned test. Once white balance and the camera matrix have been
    // applied that information is gone -- a warm highlight and a blown neutral both
    // land above 1.0 and cannot be told apart.
    let mut developer = RawDevelop::default();

    // Take over calibration so the clip test can happen before white balance.
    let cam_to_srgb = if apply_calibration {
        build_cam_to_srgb(&raw_image)
    } else {
        None
    };
    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && step != ProcessingStep::Calibrate
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer
            .steps
            .retain(|&step| step != ProcessingStep::SRgb && step != ProcessingStep::Calibrate);
    } else {
        developer
            .steps
            .retain(|&step| step != ProcessingStep::SRgb && step != ProcessingStep::Calibrate);
    }

    raw_image.wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);
    let wb_coeffs = if raw_image.wb_coeffs[0].is_nan() || !apply_calibration {
        [1.0f32, 1.0, 1.0]
    } else {
        [
            raw_image.wb_coeffs[0],
            raw_image.wb_coeffs[1],
            raw_image.wb_coeffs[2],
        ]
    };

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let safe_highlight_compression = highlight_compression.max(1.01);

    let clamp_limit = if fast_demosaic {
        1.0
    } else {
        safe_highlight_compression.max(RECONSTRUCTION_CEILING)
    };

    let max_wb = wb_coeffs[0].max(wb_coeffs[1]).max(wb_coeffs[2]);

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut linear_val = *p;
                if is_linear_format && apply_ungamma {
                    linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                }
                *p = linear_val.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut r = p[0].max(0.0);
                let mut g = p[1].max(0.0);
                let mut b = p[2].max(0.0);

                if is_linear_format && apply_ungamma {
                    r = srgb_to_linear(r.clamp(0.0, 1.0));
                    g = srgb_to_linear(g.clamp(0.0, 1.0));
                    b = srgb_to_linear(b.clamp(0.0, 1.0));
                }

                // Clip test in camera space, where every channel shares one ceiling.
                let clip_r = clippedness(r);
                let clip_g = clippedness(g);
                let clip_b = clippedness(b);

                let mut wr = r * wb_coeffs[0];
                let mut wg = g * wb_coeffs[1];
                let mut wb_ = b * wb_coeffs[2];

                // A clipped channel only tells us the scene was at least this bright.
                // The more channels have hit the ceiling the less colour information
                // survives, so blend toward the brightest channel -- a pixel with all
                // three clipped is reconstructed as neutral, which is what a blown
                // highlight actually is. One clipped channel barely moves, so warm
                // highlights that merely graze the ceiling keep their colour.
                let clipped = (clip_r + clip_g + clip_b) / 3.0;
                if clipped > 0.0 {
                    let weight = clipped * clipped;
                    let target = (wr.max(wg).max(wb_)).max(max_wb * clipped);
                    wr += (target - wr) * weight * clip_r;
                    wg += (target - wg) * weight * clip_g;
                    wb_ += (target - wb_) * weight * clip_b;
                }

                let (out_r, out_g, out_b) = match cam_to_srgb {
                    Some(m) => (
                        m[0][0] * wr + m[0][1] * wg + m[0][2] * wb_,
                        m[1][0] * wr + m[1][1] * wg + m[1][2] * wb_,
                        m[2][0] * wr + m[2][1] * wg + m[2][2] * wb_,
                    ),
                    None => (wr, wg, wb_),
                };

                p[0] = out_r.clamp(0.0, clamp_limit);
                p[1] = out_g.clamp(0.0, clamp_limit);
                p[2] = out_b.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let mut linear_val = *c;
                    if is_linear_format && apply_ungamma {
                        linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                    }
                    *c = linear_val.clamp(0.0, clamp_limit);
                });
            });
        }
    }

    let (width, height) = {
        let dim = developed_intermediate.dim();
        (dim.w as u32, dim.h as u32)
    };

    check_cancel()?;

    let dynamic_image = match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p[0], p[1], p[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p, p, p, 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        _ => {
            return Err(anyhow!("Unsupported intermediate format for conversion"));
        }
    };

    Ok((dynamic_image, orientation))
}

pub fn get_fast_demosaic_scale_factor(
    file_bytes: &[u8],
    decoded_width: u32,
    decoded_height: u32,
) -> f32 {
    let source = RawSource::new_from_slice(file_bytes);
    if let Ok(decoder) = rawler::get_decoder(&source)
        && let Ok(raw_img) = decoder.raw_image(&source, &RawDecodeParams::default(), true)
    {
        let max_orig = (raw_img.width as f32).max(raw_img.height as f32);
        let max_comp = (decoded_width as f32).max(decoded_height as f32);
        if max_orig > 0.0 {
            let ratio = max_comp / max_orig;
            if ratio > 0.1 && ratio < 0.35 {
                return 0.25;
            } else if (0.35..0.75).contains(&ratio) {
                return 0.5;
            }
        }
    }
    1.0
}
