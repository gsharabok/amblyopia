import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

from matplotlib import widgets
import models
from runner import Runner

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

sender = None
data_channel = None


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        
        self.runner = Runner()
        self.is_writer_init = False
        self.temp = 0

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")

            if not self.is_writer_init:
                self.is_writer_init = True
                self.runner.init_writer(img)

            self.runner.run_detection_calibration(img)

            # Play when the ball is in front of the face
            if self.runner.user_positioned and not self.runner.user_positioned_audio:
                data_channel.send("play correct_position")
                self.runner.user_positioned_audio = True

            # Play when the user starts to wiggle
            if self.runner.user_wiggling and not self.runner.user_wiggling_audio:
                data_channel.send("play wiggle")
                self.runner.user_wiggling_audio = True
           

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame


async def index(request):
    content = open(os.path.join(ROOT, "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def tutorial(request):
    content = open(os.path.join(ROOT, "templates/tutorial.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def training(request):
    content = open(os.path.join(ROOT, "templates/training.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "static/audio/camera_connected.mp3"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        global data_channel
        data_channel = channel
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            global sender
            sender = pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/index", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/tutorial", tutorial)
    app.router.add_get("/training", training)
    app.router.add_post("/offer", offer)

    # app.router.add_static("/", path=os.path.join(ROOT, "static"), name="static")

    app.router.add_static('/css/', path='static/css',name='css')
    app.router.add_static('/js/', path='static/js',name='js')

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )



        # if self.transform == "cartoon":
        #     img = frame.to_ndarray(format="bgr24")

        #     # prepare color
        #     img_color = cv2.pyrDown(cv2.pyrDown(img))
        #     for _ in range(6):
        #         img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        #     img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        #     # prepare edges
        #     img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #     img_edges = cv2.adaptiveThreshold(
        #         cv2.medianBlur(img_edges, 7),
        #         255,
        #         cv2.ADAPTIVE_THRESH_MEAN_C,
        #         cv2.THRESH_BINARY,
        #         9,
        #         2,
        #     )
        #     img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        #     # combine color and edges
        #     img = cv2.bitwise_and(img_color, img_edges)

        #     # rebuild a VideoFrame, preserving timing information
        #     new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        #     new_frame.pts = frame.pts
        #     new_frame.time_base = frame.time_base
        #     return new_frame

        # elif self.transform == "rotate":
        #     # rotate image
        #     img = frame.to_ndarray(format="bgr24")
        #     rows, cols, _ = img.shape
        #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
        #     img = cv2.warpAffine(img, M, (cols, rows))

        #     # rebuild a VideoFrame, preserving timing information
        #     new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        #     new_frame.pts = frame.pts
        #     new_frame.time_base = frame.time_base
        #     return new_frame