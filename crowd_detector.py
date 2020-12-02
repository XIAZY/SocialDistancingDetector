from detection import get_persons

if __name__ == '__main__':
    img = cv.imread('images/younge.jpg')

    persons = get_persons(img)
