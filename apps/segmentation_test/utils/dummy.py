import time
import depthai as dai
from depthai_nodes.node.base_host_node import BaseHostNode


class DummyNode(BaseHostNode):
    """
    Measures NN FPS based on segmentation output arrival rate
    and forwards video unchanged.
    """

    def __init__(self, print_every_sec: float = 1.0):
        super().__init__()

        self._frame_count = 0
        self._last_ts = time.perf_counter()
        self._print_every = print_every_sec

    def build(self, seg):
        self.link_args(seg)
        return self

    def process(self, seg_msg):
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_ts

        if elapsed >= self._print_every:
            fps = self._frame_count / elapsed
            print(f"NN FPS: {fps:.2f}")

            self._frame_count = 0
            self._last_ts = now


class DummyForwardNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()

    def build(self, input_msg):
        self.link_args(input_msg)
        return self

    def process(self, input_msg):
        out_msg = dai.ImgFrame()
        self.out.send(out_msg)
