# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: add aditional variables to store the current state of the controller
        self.prev_error = 0
        self.integral = 0

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        error = commanded_variable - sensor_readings[0]
        self.integral += error * self.sensor_period

        # Derivative
        derivative = (error - self.prev_error) / self.sensor_period

        pid_output = (
            self.gain_prop * error +
            self.gain_int * self.integral +
            self.gain_der * derivative
        )

        self.prev_error = error
        return pid_output