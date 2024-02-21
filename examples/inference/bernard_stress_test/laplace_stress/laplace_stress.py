# readme：代码说明参考文档https://bytedance.larkoffice.com/wiki/MCRSwW0Y3ivPHrkmz6EcpYMenRh
import argparse
import time
from concurrent import futures
from queue import Queue
from random import random, randint

from laplace import Client
from ratelimiter import RateLimiter


def stress_test(qps, period, address, model_name, duration, timeout, request_pool, batch=1):
    client = Client(address, timeout=timeout)
    rate_limiter = RateLimiter(max_calls=qps, period=period)
    pool = futures.ThreadPoolExecutor(max_workers=qps * (timeout + 2))

    # build request
    # todo: batching not implemented
    request_pool = request_pool
    # end

    time_end = time.time() + int(duration)  # run for serveral seconds
    q = Queue()

    def predict(request_pool):
        time.sleep(period * random())  # sleep for 0 ~ {period} second
        request_poolsize = len(request_pool)
        start = time.time()
        try:
            request = request_pool[randint(0, request_poolsize - 1)]
            output = client.matx_inference(model_name=model_name, input_lists=request)  # 调用请求
            msg = output.BaseResp.StatusCode
        except Exception as e:
            msg = e.__class__.__name__
            msg += " [" + str(e) + "]"

        latency = time.time() - start  # in second
        return {
            "latency": latency,
            "msg": msg,
        }

    while time.time() < time_end:
        with rate_limiter:
            q.put(pool.submit(predict, request_pool))
            continue
    time.sleep(max(timeout, 5))
    print("Sleeping...")
    time.sleep(2)
    print("Size of Result Queue: ", q.qsize())

    stats = dict()
    idx = 0
    success_count = error_count = 0
    success_latency_sum = error_latency_sum = 0

    while not q.empty():
        result = q.get().result()
        latency = result["latency"]
        msg = result["msg"]
        stats[msg] = stats.get(msg, 0) + 1

        if msg == 0:
            success_count += 1
            success_latency_sum += latency
        else:
            error_count += 1
            error_latency_sum += latency

        # report exceptions with idx
        if type(msg) == str:
            print("Exception in response", idx, ":")
            print(msg)
        idx += 1

    print("============================== Summary ==============================")
    avg_latency = (success_latency_sum + error_latency_sum) / (success_count + error_count)
    avg_latency = int(avg_latency * 1000)
    print("Avg Latency = ", avg_latency, "ms\n")

    if (success_count > 0):
        avg_success_latency = success_latency_sum / success_count
        avg_success_latency = int(avg_success_latency * 1000)
        print("Success QPS = ", float(success_count) / duration)
        print("Avg Success Latency = ", avg_success_latency, "ms\n")
    else:
        print("No success response")

    if (error_count > 0):
        avg_error_latency = error_latency_sum / error_count
        avg_error_latency = int(avg_error_latency * 1000)
        print("Error QPS = ", float(error_count) / duration)
        print("Avg Error Latency = ", avg_error_latency, "ms")
    else:
        print("No error response")

    print("============================== Distribution ==============================")
    for msg in stats:
        print("RESPONSE: ", msg if type(msg) == str else "StatusCode " + str(msg))
        print("COUNT: ", stats[msg])
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=int, default=6,
                        help="max_calls during period seconds, when period=1, this is the actual QPS of stress test")
    parser.add_argument("--period", type=int, default=1,
                        help="Real_QPS = qps / period, support for when stress test qps<1. default=1")
    parser.add_argument("--address",
                        default="xxxx",
                        help="server address")
    parser.add_argument("--model_name", default="xxxx", help="server address")
    parser.add_argument("--duration", type=int, default=120, help="Duration of stress test (in second)")
    parser.add_argument("--timeout", type=int, default=10, help="Client timeout (in second)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (#images) of each request")  # batching not implemented, keep batch=1

    # test using demo sample
    # img = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00\x00\x03\x08\x02\x00\x00\x00\xd9J"\xe8\x00\x00\x00\x12IDAT\x08\x1dcd\x80\x01F\x06\x18`d\x80\x01\x00\x00Z\x00\x04we\x03N\x00\x00\x00\x00IEND\xaeB`\x82'
    # data = {
    #     "title": ['title demo'.encode()],
    #     "desc": ['desc demo'.encode()],
    #     "ocr": ['ocr demo'.encode()],
    #     "images": [[img, img, img, img, img, img]]
    # }
    # request_pool = [data]

    # test using real samples
    import json

    data = json.load(open('/mnt/bn/chobits-wx/wx/dataset/us_sens/sens_v4/ussens_testset_1204_v2_metas.json', 'r'))
    request_pool = []
    for item in data[:10]:
        images = [open(img_path, 'rb').read() for img_path in item['images']]
        data = {
            "title": [item['title'].encode()],
            "desc": [item['desc'].encode()],
            "ocr": [item['ocr'].encode()],
            "images": [images]
        }
        request_pool.append(data)

    # do stress test
    args = parser.parse_args()
    stress_test(
        qps=args.qps,
        period=args.period,
        address=args.address,
        model_name=args.model_name,
        duration=args.duration,
        timeout=args.timeout,
        request_pool=request_pool,
        batch=args.batch
    )
