import lineTool
import requests

#六燃人流Line token:SrgflgQUZuhyoQ67F7LImPRbe1sO6iFgMIMwBQAlReJ

# Change 'token_key' to your Line token
def line_notify(msg):
    token_key = 'SrgflgQUZuhyoQ67F7LImPRbe1sO6iFgMIMwBQAlReJ'  
    header = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":'Bearer '+token_key}
    URL = 'https://notify-api.line.me/api/notify'
    payload = {'message':msg}
    res=requests.post(URL,headers=header,data=payload)
