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
import base64
from Detector import Detector
import os
import cv2


def take_screenshot_as_cv2img(driver):
    import numpy as np
    png = driver.get_screenshot_as_png()
    nparr = np.frombuffer(png, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


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

driver.find_element_by_xpath("//div[@id=\"start\"]/a").click()

while True:
    if len(driver.window_handles) > 2:
        driver.switch_to.window(driver.window_handles[2])
        break
        # driver.switch_to_window()
    print("sleep...")
    time.sleep(1)
wait = WebDriverWait(driver, 10)
cxn = wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@id=\"Cocos2dGameContainer\"]")))

while True:
    img = take_screenshot_as_cv2img(driver)
    roundId = Detector.get_round_id(img)


    print("round: " + roundId)
    if roundId:
        folder = "./mydata/rounds/%s" % roundId
        if not os.path.isdir(folder):
            os.mkdir(folder)
        else:
            millis = int(round(time.time() * 1000))
            file = "%s/%s.png" % (roundId, millis)

            cv2.imwrite("%s/%s.jpg" % (folder, millis), img)
            print("saved")
    else:
        cv2.imshow("debug",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print("sleep...")
    time.sleep(5)

# print("roundId"roundId)
