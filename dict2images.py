import cv2
import numpy as np

from letters import SYMBOLS_PIXELS


def to_image(string):
    result = [[0 for j in range(7)] for i in range(7)]

    string = string.replace('.', '0')
    new = string.split('\n')

    for i in range(1, len(new) - 1):
        line = list(new[i])
        result[i - 1] = list(map(lambda x: int(x) * 255, line))

    result = np.array(result)

    return result


'''
for a in SYMBOLS_PIXELS:
    print(SYMBOLS_PIXELS[a].replace('.', ' '))
'''
'''
serge = cv2.imread('./data/basket/serge.jpg')
serge = serge[:10, :10, :]
print(serge)
cv2.imwrite('hmeh.jpg', serge)

result = to_image(SYMBOLS_PIXELS['?'])
print(result)
cv2.imwrite('hello.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


new = cv2.imread('hello.jpg', 0)
#new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
print(new)
'''

for i in SYMBOLS_PIXELS:
    result = to_image(SYMBOLS_PIXELS[i])
    filename = './data/symbols/{}.jpg'.format(i)
    print(i, filename)
    cv2.imwrite(
        filename,
        result,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

