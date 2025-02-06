import re
import binascii

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, show, draw, ion, ioff, clf, cla, close, subplot, title, xlabel, ylabel, xlim, ylim, \
    grid, pause, gcf, gca, figure, savefig, legend, subplots, tight_layout, gca, gcf

MESSAGE_TYPES = {
    "03": "Aperture Control",
    "05": "Aperture Status",
    "06": "Focus Position Status",
    "0A": "Lens Mode Change",
    "0D": "Zoom Position Update",
    "1C": "Stop Autofocus",
    "1D": "Motor Movement (absolute/relative)",
    "1F": "Autofocus Hunt",
    "2F": "Echo Request",
    "3C": "Move at Speed",
    "40": "Stabilization Control",
    "45": "Lens Firmware Info Request"
}


def split_message(source, hex_message):
    """Splits a hex message into logical sections for better interpretation."""
    byte_data = bytes.fromhex(hex_message)

    message_length = len(byte_data)
    if message_length < 6:
        return hex_message

    data_length = message_length - 9

    objektiv_dtype = np.dtype([
        ('som', np.uint8),
        ('message_length', np.uint16),
        ('message_class', np.uint8),
        ('seq_num', np.uint8),
        ('message_type', np.uint8),
        ('data', np.uint8, data_length),
        ('checksum', np.int16),
        ('eom', np.uint8)
    ], align=False)

    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    if len(byte_array) % objektiv_dtype.itemsize != 0:
        raise ValueError("Byte array length is not compatible with the structure size.")

    objective_data = byte_array.view(objektiv_dtype)
    objective_data = np.squeeze(objective_data)

    simple = objective_data.tolist()

    # Format output string for readability
    out_message = (
        f"{source} | "
        f"SOM: {hex_message[:2]} | "
        f"Length: {hex_message[4:6] + hex_message[2:4]} ({message_length}) | "
        f"Class: {hex_message[6:8]} | "
        f"Seq#: {hex_message[8:10]} | "
        f"Type: {hex_message[10:12]} | "
        f"Payload: {hex_message[12:-6]} | "
        f"Checksum: {hex_message[-4:-2] + hex_message[-6:-4]} | "
        f"EOM: {hex_message[-2:]}"
    )


    return simple, objective_data, out_message, message_length


def parse_communication_log(file_path):
    """Parses the log file and extracts messages between body and lens."""
    data = []
    pattern = re.compile(r"(Body->Lens|Lens->Body)\s([0-9A-F]+)\s(\d+)")

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                direction = match.group(1)
                message = match.group(2)
                timestamp = int(match.group(3))

                # Categorizing Body->Lens messages for logical grouping
                message_category = "Unknown"
                if direction == "Body->Lens":
                    if message.startswith("F0"):  # Initialization and commands
                        message_category = "Command"
                    elif message.startswith("F1"):  # Status requests or parameters
                        message_category = "Status Request"
                    elif message.startswith("F2"):  # Configuration and tuning
                        message_category = "Configuration"

                data.append({
                    "direction": direction,
                    "message": message,
                    "timestamp": timestamp,
                    "category": message_category
                })
    return data


def decode_message(hex_message):
    """Decodes E-Mount messages based on known protocol structure."""
    if len(hex_message) < 12:
        return "Unknown or Invalid Message"

    msg_type = hex_message[10:12]  # Extract message type (Byte 5)
    message_description = MESSAGE_TYPES.get(msg_type, f"Unknown Command ({msg_type})")

    try:
        decoded = binascii.unhexlify(hex_message).decode('utf-8', errors='ignore')
    except Exception:
        decoded = "<Binary Data>"

    # Focus position message
    if msg_type == "06" and len(hex_message) >= 8:
        focus_position = int(hex_message[4:6], 16)
        return f"Focus Position Status - Position: {focus_position}"

    # Aperture status message
    if msg_type == "05" and len(hex_message) >= 32:
        aperture_hex = hex_message[60:64] if len(hex_message) >= 64 else hex_message[-4:]
        aperture = int(aperture_hex, 16)
        return f"Aperture Status - Value: {aperture}"

    # Autofocus stop
    if msg_type == "1C":
        return "Stop Autofocus Command Received"

    # Motor movement
    if msg_type == "1D" and len(hex_message) >= 16:
        motor_position = int(hex_message[12:16], 16)
        return f"Motor Movement Command - Target Position: {motor_position}"

    # Lens mode change
    if msg_type == "0A":
        return "Lens Mode Change Command"

    # Zoom position update
    if msg_type == "0D":
        zoom_position = int(hex_message[4:6], 16)
        return f"Zoom Position Update - Position: {zoom_position}"

    # Lens stabilization control
    if msg_type == "40":
        return "Stabilization Control Command"

    # Lens firmware info request
    if msg_type == "45":
        return "Lens Firmware Information Request"

    return message_description


def group_messages_by_type(messages, message_types):
    """
    Groups messages into a multi-dimensional NumPy array based on their type.

    Parameters:
        messages (list of list): Each message is an array containing a type ID and data.
        message_types (dict): Dictionary mapping type IDs to their meanings.

    Returns:
        dict: A dictionary where keys are message types and values are NumPy arrays of messages.
    """
    grouped = {msg_type: [] for msg_type in message_types.keys()}  # Initialize empty lists

    for message in messages:
        msg_type = format(message[4],  '02x').upper() # First element is the message type
        if msg_type in grouped:
            grouped[msg_type].append(message)  # Append full message to its type group

    # Convert lists to NumPy arrays
    grouped_numpy = {key: np.array(value, dtype=object) for key, value in grouped.items() if value}

    return grouped_numpy

def process_log(file_path):
    """Processes the log and provides human-readable output."""
    parsed_data = parse_communication_log(file_path)

    array_data = np.zeros((0, len(parsed_data)), dtype= object)
    array_data = []

    for entry in parsed_data:
        direction = entry["direction"]
        message = entry["message"]
        timestamp = entry["timestamp"]
        category = entry["category"]
        decoded_message = decode_message(message)
        simple, objective_data, data_split, length = split_message(direction, message)
        simple = np.array(simple, dtype=object)
        array_data.append(objective_data.tolist())
        # array_data = np.vstack((array_data, simple))

        print(f"[{timestamp}] {direction} ({category}): {decoded_message}\n  Structured: {data_split}\n")

    grouped_message = group_messages_by_type(array_data, MESSAGE_TYPES)

    print('finished')

    return parsed_data, array_data


# Example usage
file_path = "nex7-sony-28-70mm-at-70mm.txt"
parsed_data, array_data = process_log(file_path)

data = [ i['timestamp'] for i in parsed_data]
tim = np.diff(data) /1000

print('finished')
