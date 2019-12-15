import cv2


class TrainRank:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


def crop_maximum_area_of_contour(img):
    edged = cv2.Canny(img, 30, 300)

    cnts, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximum_area = 0
    roi = (0, 0, 0, 0)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > maximum_area:
            maximum_area = area
            roi = (x, y, w, h)

    return edged[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]


def boundingRectArea(cnt):
    _, _, w, h = cv2.boundingRect(cnt)
    return w * h


def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    ranks = ['A', '2', '3', '4', '5', '6', '7',
             '8', '9', '10', 'J', 'Q', 'K', 'JO']
    # ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
    #  'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']
    for rank in ranks:

        train_ranks.append(TrainRank())
        train_ranks[i].name = rank
        filename = rank + '.jpg'

        cropped = crop_maximum_area_of_contour(cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE))
        # edged = cv2.Canny(cv2.imread(filepath + filename, cv2.IMREAD_GRAYSCALE), 30, 300)
        #
        # cnts, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # index_sort = sorted(range(len(cnts)), key=lambda i: boundingRectArea(cnts[i]), reverse=True)
        # x, y, w, h = cv2.boundingRect(cnts[index_sort[0]])
        # bgr = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        train_ranks[i].img = cropped

        i += 1

        # cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # print("area: %s" % cv2.contourArea(cnts[i]))

    return train_ranks


tranks = load_ranks("./cards/")
