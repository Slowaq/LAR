from __future__ import print_function

from robolab_turtlebot import Turtlebot, Rate

# Names buttons and events
button_names = ['B0', 'B1', 'B2']
state_names = ['RELEASED', 'PRESSED']


def button_cb(msg):
    """Button callback."""

    # msg.button stores the id of button 0:LEFT, 1:CENTER, 2:RIGHT
    button = button_names[msg.button]

    # msg.state stores the event 0:RELEASED, 1:PRESSED
    state = state_names[msg.state]

    # Print the event
    print('{} {}'.format(button, state))


def main():
    # Initialize turtlebot class
    turtle = Turtlebot()

    # Register button callback
    turtle.register_button_event_cb(button_cb)

    # Do something, the program would end otherwise
    rate = Rate(1)
    while not turtle.is_shutting_down():
        rate.sleep()


if __name__ == '__main__':
    main()
