import urllib.request
import datetime
import requests
import json
import pickle
import time
import feedparser


def main():
    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    
    search_query = 'cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:stat.ML'
    append = True
    start_idx = 50000
    max_results = 100
    n_requests = 10000  # Max number of papers: n_requests*max_results
    day_minus = -365

    target_day = (datetime.date.today() + datetime.timedelta(days=day_minus))
        
    if append:
        with open('./data/raw_title.pkl', 'rb') as f:
            title_list = pickle.load(f)
        with open('./data/raw_abst.pkl', 'rb') as f:
            abst_list = pickle.load(f)
        with open('./data/raw_link.pkl', 'rb') as f:
            link_list = pickle.load(f)
    else:
        title_list = []
        abst_list = []
        link_list = []

    print("Current number of papers: {}".format(len(title_list)))

    n_papers = 0
    break_flag = False
    for i in range(n_requests):
        query = 'search_query={0}&start={1}&max_results={2}&sortBy=lastUpdatedDate&sortOrder=descending'.format(search_query, start_idx+n_papers, max_results)
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
        
        print("Current number of papers: {}".format(n_papers+start_idx))
        if break_flag:
            break
        time.sleep(10)
        with open('./data/raw_title.pkl', 'wb') as f:
            pickle.dump(title_list, f)
        with open('./data/raw_abst.pkl', 'wb') as f:
            pickle.dump(abst_list, f)
        with open('./data/raw_link.pkl', 'wb') as f:
            pickle.dump(link_list, f)
    
    print("Total number of papers: {}".format(n_papers))

    with open('./data/raw_title.pkl', 'wb') as f:
        pickle.dump(title_list, f)
    with open('./data/raw_abst.pkl', 'wb') as f:
        pickle.dump(abst_list, f)
    with open('./data/raw_link.pkl', 'wb') as f:
        pickle.dump(link_list, f)
        
        
if __name__ == '__main__':
    main()