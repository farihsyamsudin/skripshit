from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

chrome_driver_path = "/home/spoocream/tools/chromedriver-linux64/chromedriver"

service = Service(chrome_driver_path)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Biar gak buka browser-nya
driver = webdriver.Chrome(service=service, options=options)

driver.get("https://www.google.com")
print(driver.title)

driver.quit()
