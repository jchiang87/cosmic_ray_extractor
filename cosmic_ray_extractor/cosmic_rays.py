from collections import defaultdict
import numpy as np
import pandas as pd
import lsst.afw.detection as afw_detect
import lsst.afw.math as afw_math
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
        flags = afw_math.MEANCLIP | afw_math.STDEVCLIP
        index = -1
        for handle in exposures:
            exp = handle.get()
            det = exp.getDetector()
            det_name = det.getName()
            image = exp.getMaskedImage()
            data = defaultdict(list)
            stats = afw_math.makeStatistics(image, flags)
            mean = stats.getValue(afw_math.MEANCLIP)
            threshold_value = self.frac_threshold * mean  # "VALUE:
            if self.threshold_type == "SIGMA":
                stdev = stats.getValue(afw_math.STDEVCLIP)
                threshold_value = mean + self.nsigma * stdev
            threshold = afw_detect.createThreshold(
                threshold_value, 'value')
            footprints = afw_detect.FootprintSet(image, threshold)\
                                   .getFootprints()
            # Subtract the baseline for signal extraction.
            image -= mean
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
