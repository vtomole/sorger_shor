import os
import time


class TimeTracker:
    """
    A simple timer class that tracks elapsed time and writes it to a file.
    """
    def __init__(self, output_file=None):
        self.start_time = None
        self.total_time = 0
        self.paused = False
        self.pause_start = None
        self.output_file = output_file

        if self.output_file is not None:
            dir_path = os.path.dirname(self.output_file)
            # If a directory path is provided, ensure it exists
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # Ensure the file exists
            if not os.path.exists(self.output_file):
                with open(self.output_file, "w") as f:
                    pass  # Create the file if it doesn't exist

            # Ensure the file exists before trying to write to it
            if not os.path.exists(self.output_file):
                with open(self.output_file, "w") as f:
                    pass  # Create the file if it doesn't exist


    def start(self):
        """
        Start the timer. If the timer is already running, this will resume it.
        """
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.total_time += time.time() - self.pause_start
            self.paused = False

    def pause(self):
        """
        Pause the timer.
        """
        if self.start_time is None or self.paused:
            return
        self.pause_start = time.time()
        self.paused = True

    def stop(self):
        """
        Stop the timer and write the total elapsed time to the output file.
        """
        if self.start_time is None:
            return
        if self.paused:
            self.total_time += time.time() - self.pause_start
            self.paused = False
        else:
            self.total_time += time.time() - self.start_time
        
        # Write the total elapsed time to the file
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                f.write(f"{self.total_time:.5f}\n")
    
    def reset(self):
        """
        Reset the timer.
        """
        # Reset the timer
        self.start_time = None
        self.total_time = 0


# Example usage
if __name__ == "__main__":
    tracker = TimeTracker(output_file="time_log.txt")
    tracker.start()
    time.sleep(2)  
    tracker.pause()
    time.sleep(1)  
    tracker.start()
    time.sleep(1)
    tracker.stop()
