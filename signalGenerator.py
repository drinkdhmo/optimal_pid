import autograd.numpy as np

class signalGenerator:
    '''
        This class inherits the Signal class. It is used to organize
        1 or more signals of different types: square_wave,
        sawtooth_wave, triangle_wave, random_wave.
    '''
    def __init__(self, amplitude=1, frequency=1, y_offset=0):
        '''
            amplitude - signal amplitude.  Standard deviation for random.
            frequency - signal frequency
            y_offset  - signal y-offset
        '''
        self.amplitude = amplitude
        self.frequency = frequency
        self.y_offset = y_offset

    def square(self, tt):
        if tt % (1.0/self.frequency) <= 0.5/self.frequency:
            out = self.amplitude + self.y_offset
        else:
            out = - self.amplitude + self.y_offset
        return out
        #  returns a list of length 1

    def square_batch(self, tt):
        out = np.zeros_like(tt) - self.amplitude + self.y_offset
        out[tt % (1.0/self.frequency) <= 0.5/self.frequency] = self.amplitude + self.y_offset
        return out
        #
    #
    def sawtooth(self, tt):
        tmp = tt % (0.5/self.frequency)
        out = 4*self.amplitude*self.frequency*tmp - self.amplitude + self.y_offset
        return out
        #  returns a list of length 1

    def random(self, tt):
        out = np.sqrt(self.amplitude)*np.random.rand() + self.y_offset
        return out
        #  returns a list of length 1

    def sin(self, tt):
        out = self.amplitude*np.sin(2*np.pi*self.frequency*tt) + self.y_offset
        return out
        #  returns a list of length 1
