# -*- coding: utf-8 -*-

"""
下载相关的小工具函数；
"""

import urllib
import urllib.request

"""
商品里面根据后缀来获取真实的URL，不能保证一直有效；
"""


def get_real_url(url):
    url = url.strip()
    if url.startswith("http"):
        return url
    elif url.startswith("v1"):
        url = url.split("~")[0]
        return "https://p-multimodal.byted.org/img/ecom-shop-material/{}~tplv-houlpd90xl-mm.png".format(url)
    else:
        url = url.split("~")[0]
        return "https://p-multimodal.byted.org/img/temai/{}~tplv-houlpd90xl-mm.png".format(url)


def get_original_urls(urls):
    urls_new = []
    for url in urls:
        suffix = url.split("/")[-1].split("~")[0]
        if "ecom-shop-material" in url and "p-multimodal.byted.org" in url:
            urls_new.append("https://p9-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
            urls_new.append("https://p6-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
            urls_new.append("https://p3-aio.ecombdimg.com/obj/ecom-shop-material/{}".format(suffix))
        elif "temai" in url and "p-multimodal.byted.org" in url:
            urls_new.append("https://p9-aio.ecombdimg.com/obj/temai/{}".format(suffix))
            urls_new.append("https://p6-aio.ecombdimg.com/obj/temai/{}".format(suffix))
            urls_new.append("https://p3-aio.ecombdimg.com/obj/temai/{}".format(suffix))
        urls_new.append(url)

    return urls_new


"""
从消重侧拿过来的url转换方法，用于兜底使用；
"""


def further_real_url(url):
    url = url.replace("sf1-ttcdn-tos.pstatp.com", "p-multimodal.byted.org/")
    url = url.replace("sf3-ttcdn-tos.pstatp.com", "p-multimodal.byted.org/")
    url = url.replace("sf6-ttcdn-tos.pstatp.com", "p-multimodal.byted.org/")
    url = url.replace("sf9-ttcdn-tos.pstatp.com", "p-multimodal.byted.org/")
    url = url.replace("p6-aio.ecombdimg.com", "p-multimodal.byted.org/")
    url = url.replace("p3-aio.ecombdimg.com", "p-multimodal.byted.org/")
    url = url.replace("p9-aio.ecombdimg.com", "p-multimodal.byted.org/")
    url = url.replace("tosv.byted.org", "p-multimodal.byted.org/")
    if "multimodal" in url and "/obj/" in url:
        url = url.replace("/obj/", "/img/") + "~800x800.jpg"
    return url


"""
根据url下载商品图像，如果超时1s，那么直接返回空的bytes；
"""


def download_url_with_exception(url: str, timeout=3):
    try:
        req = urllib.request.urlopen(url=url, timeout=timeout)
        return req.read()
    except:  # noqa: E722
        return b""
