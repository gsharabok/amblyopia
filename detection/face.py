import numpy as np

def detect_face(img):
        coords = img

        if len(coords) > 1:
            biggest = (0, 0, 0, 0)
            for i in coords:
                if i[3] > biggest[3]:
                    biggest = i
            # noinspection PyUnboundLocalVariable
            biggest = np.array([i], np.int32)
        elif len(coords) == 1:
            biggest = coords
        else:
            return None

        for (x, y, w, h) in biggest:
            frame = img[y : y + h, x : x + w]
            return frame


def get_largest_frame(imgs):
    if len(imgs) > 1:
        biggest = imgs[0]
        for cur in imgs:
            if cur[3] > biggest[3]:
                biggest = cur

        return [biggest]
    elif len(imgs) == 1:
        return imgs
    else:
        return imgs