pip install openpyxl

from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pytz
import pandas as pd
import openpyxl

format = 'DATE:%d-%b-%Y  TIME-%I:%M:%p'
Zone = pytz.timezone('Asia/Kolkata')
today = datetime.now(Zone)
today = today.strftime(format)

# Date Time
r = requests.get(url)
print("response status Code:",r.status_code)

#Parsing HTML
# We got status code of 2xx and we are good to go.
soup = BeautifulSoup(r.content,'html.parser')
print(soup.prettify)

print(len(soup.find_all('img')))

# We got 36 images detail but on reaching webpage we see only 32 images are with quote on it.
# we need to filter out unambigous images from our soup.
Big_table = soup.find('div',attrs={'class':'row','id':'all_quotes'})

All_image = Big_table.find_all('img') # We have filtered out all images with IDs and Class
print(len(All_image)) # thats correct number = 32

for image in All_image:
    t= image['src']
    data = data.append({'Type':'Source','URL':t},ignore_index = True)
    
data.to_excel('quotes_source_code.xlsx')



for url in data['URL']:
    req = requests.get(url)
    with open('img_quote.jpg'wb') as ft:
        ft.write(req.content)
