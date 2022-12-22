import json
import urllib.request
from bs4 import BeautifulSoup

def pachong(url):
    # response = urllib.request.urlopen(url)
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url=url, headers=headers)
    html = urllib.request.urlopen(req).read()
    # print(html)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    return soup.find('header').find('h1').get_text()

# pachong("http://mashable.com/2013/01/07/amazon-instant-video-browser/")
# exit()
urls_json = json.load(open('data/urls.json', 'r'))
for dataset in ['train', 'test']:
    data = []
    print(dataset, len(urls_json[dataset]['url']))
    for i, url in enumerate(urls_json[dataset]['url']):
        data.append(pachong(url))
        if i % 1000 == 0: print(i)
    json.dump({'articles':data, 'labels': urls_json[dataset]['label']}, open('data/{}_article.json'.format(dataset), 'w'))
