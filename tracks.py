import numpy as np


class Tracklet:
    def __init__(self, pos, track_id):
        self.id = track_id
        self.last_pos = pos
        self.pos_list = [pos]
        self.pos_num = 0
        self.last_frame = pos.t
        self.vel = [0,0]


    def add_pos(self, pos):
        self.pos_list.append(pos)
        self.pos_num += 1
        self.last_pos = pos
        self.last_frame = pos.t
        self.update_vel()

    def set_id(self, new_id):
        self.id = new_id

    def update_vel(self):
        if (self.pos_num < 2):
            return
        elif (self.pos_num < 6):
            vx = (self.pos_list[self.pos_num - 1].x - self.pos_list[self.pos_num - 2].x) / (
                        self.pos_list[self.pos_num - 1].t - self.pos_list[self.pos_num - 2].t)
            vy = (self.pos_list[self.pos_num - 1].z - self.pos_list[self.pos_num - 2].z) / (
                        self.pos_list[self.pos_num - 1].t - self.pos_list[self.pos_num - 2].t)
            self.v = [vx, vy]
        else:
            vx1 = (self.pos_list[self.pos_num - 1].x - self.pos_list[self.pos_num - 4].x) / (
                        self.pos_list[self.pos_num - 1].t - self.pos_list[self.pos_num - 4].t)
            vy1 = (self.pos_list[self.pos_num - 1].z - self.pos_list[self.pos_num - 4].z) / (
                        self.pos_list[self.pos_num - 1].t - self.pos_list[self.pos_num - 4].t)
            vx2 = (self.pos_list[self.pos_num - 2].x - self.pos_list[self.pos_num - 5].x) / (
                        self.pos_list[self.pos_num - 2].t - self.pos_list[self.pos_num - 5].t)
            vy2 = (self.pos_list[self.pos_num - 2].z - self.pos_list[self.pos_num - 5].z) / (
                        self.pos_list[self.pos_num - 2].t - self.pos_list[self.pos_num - 5].t)
            vx3 = (self.pos_list[self.pos_num - 3].x - self.pos_list[self.pos_num - 6].x) / (
                        self.pos_list[self.pos_num - 3].t - self.pos_list[self.pos_num - 6].t)
            vy3 = (self.pos_list[self.pos_num - 3].z - self.pos_list[self.pos_num - 6].z) / (
                        self.pos_list[self.pos_num - 3].t - self.pos_list[self.pos_num - 6].t)
            vx, vy = (vx1 + vx2 + vx3) / 3, (vy1 + vy2 + vy3) / 3
            self.v = [vx, vy]