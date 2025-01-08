import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

class Go2Action:
    def __init__(self):
        self.client = SportClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self.dt = 0.01
    
    def StandUp(self):
        self.client.RiseSit()
        time.sleep(1)

    """
    def StandUp(self):
        self.client.RecoveryStand()
        self.client.BalanceStand()
        time.sleep(1)
    """

    def SitDown(self):
        self.client.Sit()
        time.sleep(1)
    
    def Stretch(self):
        self.client.Stretch()
        time.sleep(1)
    
    def Dance(self):
        self.client.Dance1()
        time.sleep(1)
    
    def FrontJunmp(self):
        self.client.FrontJump()
        time.sleep(1)
    
    def Heart(self):
        self.client.Heart()
        time.sleep(1)
    
    def FrontFlip(self):
        self.client.FrontFlip()
        time.sleep(1)

    def Move(self, x, y, z):
        """x(m)、y(m)、z(rad)回転する"""
        for i in range(int(x / self.dt)):
            self.client.Move(1, 0, 0)
            time.sleep(self.dt)
        self.clinet.StopMove()

        for i in range(int(y / self.dt)):
            self.client.Move(0, 1, 0)
            time.sleep(self.dt)
        self.clinet.StopMove()

        for i in range(int(z / self.dt)):
            self.client.Move(0, 0, 1)
            time.sleep(self.dt)
        self.clinet.StopMove()

        time.sleep(1)