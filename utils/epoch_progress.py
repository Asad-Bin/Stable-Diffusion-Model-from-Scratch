import sys

def print_epoch_progress(current_epoch, total_epochs, bar_length=30):
    # Calculate progress fraction
    fraction = current_epoch / total_epochs
    # Number of bar characters to show
    filled_length = int(bar_length * fraction)

    # Create the bar string
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)
    # Print the bar with carriage return to overwrite the same line
    sys.stdout.write(f"\rEpoch {current_epoch}/{total_epochs} [{bar}]")
    sys.stdout.flush()
