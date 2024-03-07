import numpy as np


class SquatsCounter:
    def __init__(self, str_leg_angle=80, str_body_angle=110):
        """Class for counting squats

        Parameters
        ----------
        str_leg_angle : int, optional
            Threshold angle in legs in stressed state, by default 110
        str_body_angle : int, optional
            Threshold angle in body in stressed state, by default 100
        """
        # thresholds
        self.str_leg_angle = str_leg_angle
        self.str_body_angle = str_body_angle
        # flags
        self.stressed = False
        self.relaxed = False
        self.deep = False
        self.count = 0

    def reset(self) -> None:
        """Reset state of counter
        """
        self.count = 0
        self.stressed = False
        self.relaxed = False

    def step(self, leg_angle: int, body_angle: int) -> tuple:
        """Update step of counter 

        Parameters
        ----------
        leg_angle : int
            Current angle in leg
        str_body_angle : int
            Current angle in body

        Returns
        -------
        tuple
            Legs and body state
        """
        legs = False  # stress indicators
        body = False
        if self.str_leg_angle - 10 < leg_angle < self.str_leg_angle + 10:
            legs = True
        if self.str_body_angle - 10 < body_angle < self.str_body_angle + 10:
            body = True
        if legs and body:
            self.stressed = True
        if leg_angle < self.str_leg_angle - 30:
            self.deep = True
        if 160 < body_angle < 180 and 160 < leg_angle < 180 and self.stressed:
            self.relaxed = True
        if self.stressed and self.relaxed:
            if not self.deep: 
                self.count += 1
            self.stressed = False
            self.relaxed = False
            self.deep = False
        return legs, body

    def get_count(self) -> int:
        """Get number of counts

        Returns
        -------
        int
            Current number of counts
        """
        return self.count
