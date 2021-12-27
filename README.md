# arXiv-search-local
タイトルとアブストラクトの情報から類似するarXivの論文を検索するツールです。

## requirements
- torch
- numpy 
- transformers
- requests
- feedparser==5.1.1

Ubuntu Server 16.04, Python3.8で動作確認済み

## Run
まず、arXivのAPIを利用して論文データをダウンロードします。
```
python get-papers.py --query cat:cs.LG  --start-idx 0 --max-results 100 --day-minus 10 --n-requests 100 (--append)
```
各argsの詳細は以下の通りです（全てoptional）。公式ドキュメントの[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)も参考にしてください。

| --option | 説明 | デフォルト値 |
|-|-|-|
| --query | arXiv APIのクエリ（[詳細](https://arxiv.org/help/api/user-manual#query_details)）。例えば、`cat:cs.AI+OR+cat:cs.CV`と指定すればcs.AIまたはcs.CVのカテゴリに含まれる論文をダウンロードできます | cat:cs.LG |
| --day-minus | 何日前までの論文をダウンロードするかを示す。全期間の論文をダウンロードするには10000などと指定してください。 | 10000 |
| --append | 今回ダウンロードした論文データを前回までにダウンロードしたデータに付加するかどうかを示す。--appendを付けなければ上書きされます。 | 10000 |
| --max-results | 1回のリクエストでダウンロードする論文数の最大値。10000以下の値に設定してください。値が大きいとリクエストに繰り返し失敗することがあるので適宜調整してください。詳しくは[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)を参照してください。 | 10000 |
| --start-idx | 何番目の論文からダウンロードするかを表すindex。詳しくは[Query Interface](https://arxiv.org/help/api/user-manual#_query_interface)を参照してください。 | 0 |
