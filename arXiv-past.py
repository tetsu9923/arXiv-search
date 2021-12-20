import urllib.request
import feedparser
import datetime
import requests
import json
import pickle
import time


def main():
    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    
    search_query = 'cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:stat.ML'
    max_results = 5000
    n_requests = 10  # Max number of papers: n_requests*max_results
    day_minus = -50

    target_day = (datetime.date.today() + datetime.timedelta(days=day_minus))
        
    title_list = []
    abst_list = []
    link_list = []
    n_papers = 0
    break_flag = False
    for i in range(n_requests):
        query = 'search_query={0}&start={1}&max_results={2}&sortBy=lastUpdatedDate&sortOrder=descending'.format(search_query, max_results*i, max_results)
        # feedparser v4.1
        feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
        feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'
        
        # perform a GET request using the base_url and query
        with urllib.request.urlopen(base_url + query) as url:
            response = url.read()
            
        # parse the response using feedparser
        feed = feedparser.parse(response)
        
        for entry in feed.entries:
            published_date = (entry.updated.split('T'))[0]
            published_date = datetime.datetime.strptime(published_date, "%Y-%m-%d").date()
            if published_date < target_day:
                break_flag = True
                break
            title_list.append(entry.title)
            abst_list.append(entry.summary.replace("\n", " "))
            link_list.append(entry.links[0]['href'])
            n_papers += 1
        
        print(n_papers)
        if break_flag:
            break
        time.sleep(10)
    
    print("Total number of papers: {}".format(n_papers))
    return title_list, abst_list, link_list
        
        
if __name__ == '__main__':
    title_list, abst_list, link_list = main()
    with open('./data/raw_title.pkl', 'wb') as f:
        pickle.dump(title_list, f)
    with open('./data/raw_abst.pkl', 'wb') as f:
        pickle.dump(abst_list, f)
    with open('./data/raw_link.pkl', 'wb') as f:
        pickle.dump(link_list, f)