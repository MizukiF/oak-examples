from dotenv import load_dotenv
import os  # noqa: F401
import time

# os.environ.setdefault("DEPTHAI_LEVEL", "INFO")
# os.environ.setdefault("DEPTHAI_NODES_LEVEL", "INFO")
import depthai as dai

from depthai_nodes.node import ParsingNeuralNetwork
from utils.arguments import initialize_argparser

from utils.dummy import DummyNode, DummyForwardNode  # noqa: F401


load_dotenv(override=True)
_, args = initialize_argparser()

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatformAsString()
print(f"Platform: {platform}")

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    cam = pipeline.create(dai.node.Camera).build()

    video_full = cam.requestOutput(
        size=(512, 288),
        type=dai.ImgFrame.Type.BGR888i,
        fps=30,
    )

    model = dai.NNModelDescription("luxonis/fastsam-s:512x288")
    model.platform = platform
    archive = dai.NNArchive(dai.getModelFromZoo(model))
    w, h = archive.getInputSize()

    nn_with_parser_node = pipeline.create(ParsingNeuralNetwork).build(
        video_full,
        archive,
    )  # ~19 FPS

    # nn_node = pipeline.create(dai.node.NeuralNetwork).build(
    #     video_full,
    #     archive,
    # )  # by itself bottlenecked by input frame rate, raw benchmark 490 FPS

    dummy_node = pipeline.create(DummyNode).build(
        nn_with_parser_node.out,
        video_full,
    )  # ~19FPS because there is device -> host transfer of NN output

    # dummy_forward_node = pipeline.create(
    #     DummyForwardNode
    # ).build(
    #     nn_with_parser_node.out
    # )  # ~19FPS - No additional transfer so FPS is "same" as with just ParsingNeuralNetwork

    benchmarkIn = pipeline.create(dai.node.BenchmarkIn)
    benchmarkIn.logReportsAsWarnings(True)
    benchmarkIn.sendReportEveryNMessages(10)
    # nn_node.out.link(benchmarkIn.input)
    # dummy_forward_node.out.link(benchmarkIn.input)
    dummy_node.out.link(benchmarkIn.input)

    print("Pipeline created.")

    pipeline.start()

    while pipeline.isRunning():
        time.sleep(0.001)
