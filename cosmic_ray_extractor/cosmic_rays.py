from collections import defaultdict
import numpy as np
import pandas as pd
import lsst.afw.math as afw_math
import lsst.meas.algorithms as meas_alg
import lsst.pex.config as pex_config
import lsst.pipe.base as pipe_base
from lsst.pipe.base import connectionTypes as cT


__all__ = ("CosmicRaysTask",)


class CosmicRaysTaskConnections(
        pipe_base.PipelineTaskConnections,
        dimensions=("instrument", "detector")
):
    """Task to extract cosmic rays from ISR'd dark frames."""

    exposures = cT.Input(
        name="darkIsr",
        doc="ISR'd dark frames",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )

    cosmic_ray_catalog = cT.Output(
        name="cosmic_ray_catalog",
        doc="Catalog of cosmic-rays from dark frames.",
        storageClass="DataFrame",
        dimensions=("instrument", "detector"),
    )


class CosmicRaysTaskConfig(
        pipe_base.PipelineTaskConfig,
        pipelineConnections=CosmicRaysTaskConnections
):
    psf_size = pex_config.Field(
        dtype=int,
        doc="PSF size for CR detection in pixels",
        default=21,
    )

    psf_fwhm = pex_config.Field(
        dtype=float,
        doc="PSF FWHM for CR detection in pixels",
        default=3.0,
    )

    background_size = pex_config.Field(
        doc="Approximate size in pixels of cells used for background scaling.",
        dtype=int, default=64
    )
    threshold_type = pex_config.ChoiceField(
        dtype=str,
        doc="Type of detection threshold.",
        default="SIGMA",
        allowed={"VALUE": "Use fractional_threshold * clipped mean.",
                 "SIGMA": "Use nsigma * clipped stdev + clipped mean."}
    )
    fractional_threshold = pex_config.Field(
        doc=("Fractional threshold above clipped mean for"
             "cosmic-ray pixel detection."),
        dtype=float, default=1.2
    )
    nsigma = pex_config.Field(
        doc=("Number of clipped stdev above clipped mean for "
             "cosmic-ray pixel detection."),
        dtype=float, default=5.0
    )


class CosmicRaysTask(pipe_base.PipelineTask):
    """
    Class for detecting and extracting cosmic-rays from dark frames.
    """
    ConfigClass = CosmicRaysTaskConfig
    _DefaultName = "cosmicRaysTask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg_size = self.config.background_size
        self.threshold_type = self.config.threshold_type
        self.frac_threshold = self.config.fractional_threshold
        self.nsigma = self.config.nsigma

    def run(self, exposures):
        # See https://github.com/lsst/pipe_tasks/blob/w.2026.12/python/lsst/pipe/tasks/repair.py#L226 and https://github.com/lsst/cp_pipe/blob/w.2026.12/python/lsst/cp/pipe/cpDark.py#L118
        psf = meas_alg.SingleGaussianPsf(
            self.config.psf_size,
            self.config.psf_size,
            self.config.psf_fwhm/(2.*np.sqrt(2.*np.log(2)))
        )
        cr_config = pex_config.makePropertySet(
            meas_alg.findCosmicRaysConfig.FindCosmicRaysConfig()
        )

        index = -1
        data = defaultdict(list)
        for handle in exposures:
            exp = handle.get()
            exp.setPsf(psf)
            det = exp.getDetector()
            det_name = det.getName()
            image = exp.getMaskedImage()
            stats = afw_math.makeStatistics(image, afw_math.MEDIAN)
            median_bg = stats.getValue(afw_math.MEDIAN)
            keep_crs = True
            footprints = meas_alg.findCosmicRays(
                image,
                psf,
                median_bg,
                cr_config,
                keep_crs
            )
            # Subtract the baseline for signal extraction.
            image -= median_bg
            for fp in footprints:
                index += 1
                for span in fp.getSpans():
                    data['index'].append(index)
                    data['exposure'].append(handle.dataId['exposure'])
                    data['det_name'].append(det_name)
                    iy = span.getY()
                    ix0 = span.getX0()
                    ix1 = span.getX1()
                    row = image.getImage().array[iy, ix0:ix1 + 1]
                    data['x0'].append(ix0)
                    data['y0'].append(iy)
                    data['pixel_values'].append(np.array(row, dtype=int))
        return pipe_base.Struct(
            cosmic_ray_catalog=pd.DataFrame(data)
        )

    def _background_scaling(self, image):
        bbox = image.getBBox()
        nx = bbox.getWidth() // self.bg_size
        ny = bbox.getHeight() // self.bg_size
        bg_ctrl = afw_math.BackgroundControl(nx, ny)
        bg_obj = afw_math.makeBackground(image, bg_ctrl)
        image /= bg_obj.getImageF("LINEAR")
        return image
