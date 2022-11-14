import requests
from bs4 import BeautifulSoup

URL = "https://digitalcollections.nypl.org/collections/childrens-book-illustrations?filters[rights]=pd&keywords=#/?tab=navigation&roots=23:6a98be10-c5ba-012f-3233-58d385a7bc34"

page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
print(soup)
results = soup.find(id="collection-right")
results = results.find(id="results-list")
results = results.find_all("li")
i = 0
max_iter = 1
for r in results:
    i+=1
    if i == 10:
        break
    print(r)
    #raise Exception
    div_item = r.find_all("div", class_="item")[0]

    div_desc = r.find_all("div", class_="description")[0]

    img_url = div_item.find_all("img")[0]["src"]
    print(div_desc)
    description = div_desc.find_all("a")[0].text
    print("\n",img_url, description)



    ## Save image
    img_data = requests.get(img_url).content
    with open("/home/torstein/src/stanford/cs229/StableDiffusionProject/web-scraping/data/"+description+".jpg", "wb") as handler:
        handler.write(img_data)

    #title = div.find("a").title()
#print(results.prettify())

