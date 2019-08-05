class Const():
    DOOR_Y_MIN = 600
    DOOR_X_MIN = 700
    DOOR_Y_MAX = 1500
    DOOR_X_MAX = 900

    RESOLUTION_WIDTH = 1
    RESOLUTION_HEIGHT = 1

    @staticmethod
    def set_resolution(width, height):
        Const.RESOLUTION_WIDTH = width
        Const.RESOLUTION_HEIGHT = height
        Const.DOOR_X_MAX = height

    @staticmethod
    def get_start():
        return Const.DOOR_Y_MIN, Const.DOOR_X_MIN

    @staticmethod
    def get_end():
        return Const.DOOR_Y_MAX, Const.DOOR_X_MAX

    @staticmethod
    def get_start_ratio():
        return Const.DOOR_Y_MIN/Const.RESOLUTION_WIDTH, Const.DOOR_X_MIN/Const.RESOLUTION_HEIGHT

    @staticmethod
    def get_end_ratio():
        return Const.DOOR_Y_MAX/Const.RESOLUTION_WIDTH, 1.0

