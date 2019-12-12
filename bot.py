from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
import credentials as cred
driver = webdriver.Firefox()
# driver = webdriver.Firefox(executable_path='./geckodriver.exe')
driver.get("https://www.gamesofa.com/landlord/?op=login")

driver.find_element_by_id("username").send_keys(cred.username)

driver.find_element_by_id("password").send_keys(cred.password)
driver.find_element_by_id("login").click()
driver.find_element_by_class_name("landlord").click()

while True:
    if len(driver.window_handles) > 1:
        driver.switch_to.window(driver.window_handles[1])
        break
        # driver.switch_to_window()
    time.sleep(1)

#
# wait = WebDriverWait(driver, 100)
# startBtn = wait.until(ec.h((By.ID, "start")))
# ActionChains(driver).move_to_element(startBtn).perform()



import base64

import cv2

cv2.imshow(base64.b64decode(driver.get_screenshot_as_base64()))
cv2.imdecode()
driver.find_element_by_xpath("//div[@id=\"start\"]/a").click()
# while True:
#     if len(driver.window_handles) > 2:
#         driver.switch_to.window(driver.window_handles[2])
#         break
#         # driver.switch_to_window()
#     print("sleep...")
#     time.sleep(1)
# wait = WebDriverWait(driver, 10)
# cxn = wait.until(ec.visibility_of_element_located((By.XPATH, "//div[id=\"Cocos2dGameContainer\"]")))
#
