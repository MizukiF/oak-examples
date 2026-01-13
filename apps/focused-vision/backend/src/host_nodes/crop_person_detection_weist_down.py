from typing import Callable

import depthai as dai


class CropPersonDetectionWaistDown(dai.node.HostNode):
    """"""

    def __init__(self) -> None:
        super().__init__()
        self._ymin_transformer = None
        self._ymax_transformer = None

    def build(
        self,
        nn_output: dai.Node.Output,
        ymin_transformer: Callable[[dai.ImgDetection], float],
        ymax_transformer: Callable[[dai.ImgDetection], float],
    ) -> "CropPersonDetectionWaistDown":
        self.link_args(nn_output)
        self._ymin_transformer = ymin_transformer
        self._ymax_transformer = ymax_transformer
        # self.sendProcessingToPipeline(False)
        return self

    def process(self, nn_output: dai.Buffer) -> None:
        assert isinstance(nn_output, dai.ImgDetections)
        for detection in nn_output.detections:
            detection.ymax = self._ymax_transformer(detection)
            detection.ymin = self._ymin_transformer(detection)
        self.out.send(nn_output)
