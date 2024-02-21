import base64
import io
import urllib
from urllib import request

from PIL import Image


def get_real_url(url):
    url = url.strip()
    real_url = f"https://p16-oec-va.ibyteimg.com/{url}~512x512.jpg"

    return real_url


def get_original_url(uri):
    uri = uri.strip()
    if uri.startswith("/obj/"):
        original_url = f"https://p16-ecomcdn-va.ibyteimg.com{uri}"
    else:
        original_url = f"https://p16-oec-va.ibyteimg.com/{uri}~tplv-o3syd03w52-origin-jpeg.jpeg"

    return original_url


def download_url_with_exception(url: str, timeout=2):
    try:
        req = urllib.request.urlopen(url=url, timeout=timeout)
        return req.read()
    except:
        return b""


def download_image_to_base64(url: str, timeout=2, rt="str"):
    # bytes
    if rt == "bytes":
        try:
            req = urllib.request.urlopen(url=url, timeout=timeout)
            return base64.b64encode(req.read())
        except:
            return b""

    elif rt == "str":
        try:
            req = urllib.request.urlopen(url=url, timeout=timeout)
            return base64.b64encode(req.read()).decode("utf-8")
        except:
            return ""

    else:
        raise Exception(f"only support str and bytes")


def get_single_image_from_urls(urls: list, load=False):
    # return image before transform
    try:
        image_str = b""
        for url in urls:
            url = get_real_url(url)
            image_str = download_url_with_exception(url, timeout=3)
            if image_str != b"" and image_str != "":
                break
            else:
                print(f"Empty image: {url}")
    except:
        image_str = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xdb\x00C\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x11\x08\x00\x02\x00\x02\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x15\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xbf\x80\x01\xff\xd9'

    if load:
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        return image
    else:
        return image_str


# https://p16-oec-va.ibyteimg.com/tos-maliva-i-o3syd03w52-us/f7d22e50c5ed4eb8b77ef296af178ac3~tplv-o3syd03w52-resize-jpeg:200:200.image?
# [{"uri":"tos-alisg-i-aphluv4xwc-sg/017ffdb17e314797bee5252221b62550","height":1282,"width":1282},
# {"uri":"tos-alisg-i-aphluv4xwc-sg/d744aa57cc49430b997d7edfca385e90","height":1280,"width":1280}]

# def get_original_urls(urls):
#     urls_new = []
#     for url in urls:
#         suffix = url.split("/")[-1].split("~")[0]
#         if "ecom-shop-material" in url and "p-multimodal.byted.org" in url:
#             urls_new.append("https://p9-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
#             urls_new.append("https://p6-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
#             urls_new.append("https://p3-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
#         elif "temai" in url and "p-multimodal.byted.org" in url:
#             urls_new.append("https://p9-aio.ecombdimg.com/obj/temai/{}".format(suffix))
#             urls_new.append("https://p6-aio.ecombdimg.com/obj/temai/{}".format(suffix))
#             urls_new.append("https://p3-aio.ecombdimg.com/obj/temai/{}".format(suffix))
#         urls_new.append(url)
#
#     return urls_new


# def further_real_url(url):
#     """
#     从消重侧拿过来的url转换方法，用于兜底使用；
#     """
#     url = url.replace('sf1-ttcdn-tos.pstatp.com', 'p-multimodal.byted.org/')
#     url = url.replace('sf3-ttcdn-tos.pstatp.com', 'p-multimodal.byted.org/')
#     url = url.replace('sf6-ttcdn-tos.pstatp.com', 'p-multimodal.byted.org/')
#     url = url.replace('sf9-ttcdn-tos.pstatp.com', 'p-multimodal.byted.org/')
#     url = url.replace('p6-aio.ecombdimg.com', 'p-multimodal.byted.org/')
#     url = url.replace('p3-aio.ecombdimg.com', 'p-multimodal.byted.org/')
#     url = url.replace('p9-aio.ecombdimg.com', 'p-multimodal.byted.org/')
#     url = url.replace('tosv.byted.org', 'p-multimodal.byted.org/')
#     if 'multimodal' in url and '/obj/' in url:
#         url = url.replace('/obj/', '/img/') + '~800x800.jpg'
#     return url


if __name__ == "__main__":
    s = download_url_with_exception(
        "https://p16-oec-va.ibyteimg.com/tos-useast2a-i-hyqnpo4tzp-aiso/74d7e36be674463eb3202872e343198d~800x800.jpg"
    )
    image = Image.open(io.BytesIO(s)).convert("RGB")
    # print(image)
    from torchvision_dataset import get_transform

    form = get_transform()
    print(form(image))
