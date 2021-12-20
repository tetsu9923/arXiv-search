import urllib.request
import feedparser
import datetime
import requests
import json
import pickle


def main():
    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Search parameters
    # search_query = 'cat:cs.CV OR cat:cs.MM'
    search_query = 'cat:cs.AI+OR+cat:cs.LG'
    start = 0
    max_results = 1000
    n_papers = 0
    
    query = 'search_query={0}&start={1}&max_results={2}&sortBy=lastUpdatedDate&sortOrder=descending'.format(search_query, start, max_results)
    
    today_date_index = datetime.date.today().weekday() # 0(Mon)~6(Sun)
    text = ""
    
    # get yesterday's update
    # (Note: updates on Friday and Saturday are updated on Monday)
    if today_date_index == 0:  # Mon
        days_minus_to_search = [-3, -2, -1]
    elif today_date_index == 5 or today_date_index == 6:  # Sat, Sun
        days_minus_to_search = []
    else:
        days_minus_to_search = [-1]
        
    target_days = []
    title_list = []
    abst_list = []
    link_list = []
    for day_minus in days_minus_to_search:  # Not exexuted when Sat and Sun
        target_day = (datetime.date.today() + datetime.timedelta(days=day_minus)).strftime("%Y-%m-%d")
        target_days.append(target_day)
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
            if not published_date == target_day:
                continue
            title_list.append(entry.title)
            abst_list.append(entry.summary)
            link_list.append(entry.links[0]['href'])
            n_papers += 1
    
    return title_list, abst_list, link_list
        
        
if __name__ == '__main__':
    title_list, abst_list, link_list = main()
    with open('./data/raw_title.pkl', 'wb') as f:
        pickle.dump(title_list, f)
    with open('./data/raw_abst.pkl', 'wb') as f:
        pickle.dump(abst_list, f)
    with open('./data/raw_link.pkl', 'wb') as f:
        pickle.dump(link_list, f)