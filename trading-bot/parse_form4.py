import requests, json, time
import xml.etree.ElementTree as ET
UA={"User-Agent":"kadri-research research@example.com"}
tickers={"AMD":"0000002488","AAPL":"0000320193","GOOGL":"0001652044","MSFT":"0000789019",
         "NVDA":"0001045810","TSLA":"0001318605","META":"0001326801","AMZN":"0001018724"}
PER=70
out=[]; summary={}
for t,cik in tickers.items():
    try:
        j=requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",headers=UA,timeout=30).json()
    except Exception as e:
        summary[t]=f"submissions failed: {e}"; continue
    r=j["filings"]["recent"]; codes={}; n=0; buys=0
    for i in range(len(r["form"])):
        if r["form"][i]!="4": continue
        acc=r["accessionNumber"][i].replace("-",""); doc=r["primaryDocument"][i].split("/")[-1]
        url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        try:
            root=ET.fromstring(requests.get(url,headers=UA,timeout=20).text)
        except Exception:
            continue
        for tx in root.iter("nonDerivativeTransaction"):
            ce=tx.find(".//transactionCoding/transactionCode")
            if ce is None: continue
            codes[ce.text]=codes.get(ce.text,0)+1
            if ce.text=="P":
                sh=tx.find(".//transactionShares/value"); pr=tx.find(".//transactionPricePerShare/value")
                buys+=1
                out.append({"ticker":t,"date":r["filingDate"][i],
                            "shares":sh.text if sh is not None else None,
                            "price":pr.text if pr is not None else None})
        n+=1
        if n>=PER: break
    summary[t]={"parsed":n,"purchases":buys,"codes":codes}
    print(f"{t}: parsed {n}, purchases {buys}, codes {codes}", flush=True)
json.dump({"summary":summary,"purchases":out}, open("form4_buys.json","w"), indent=2)
print("DONE")
