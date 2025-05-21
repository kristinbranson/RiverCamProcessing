# set up environment
import cv2
import numpy as np
import os
import sys
import json
import tqdm
import re

def resize_frame(frame, imwidth):
    """resize_frame(frame, imwidth) -> frame
    Resize the frame to have width imwidth, keeping the aspect ratio.
    """
    height, width, _ = frame.shape
    if width > imwidth:
        scale = imwidth / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    return frame

def recenter_frame(frame,x0):
    if x0 == 0:
        return frame
    width, height = frame.shape[1], frame.shape[0]
    outframe = np.concatenate((frame[:, x0:], frame[:, :x0]), axis=1)
    return outframe

def get_video_info(moviefile=None,cap=None):
    """
    Get video information such as width, height, and frames per second (fps).
    
    Parameters:
    - moviefile: Path to the input video file.
    
    Returns:
    - width: Width of the video.
    - height: Height of the video.
    - fps: Frames per second of the video.
    """
    
    # open video file
    didopen = False
    if cap is None:
        cap = cv2.VideoCapture(moviefile)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None
        didopen = True
        
    # get video properties
    info = {}
    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
    info['num_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if didopen:
        # close video file 
        cap.release()
    
    return info

def get_cropvideo_name(moviefile,c0,c1,r0,r1):
    newext = '.avi'
    # split input file name into name and extension
    basename,ext = os.path.splitext(moviefile)
    outfile = basename + f'_r{r0}to{r1}_c{c0}to{c1}{newext}'
    return outfile

def crop_video(**kwargs):
    """
    Crop a video file into multiple regions and save them as separate files.
    
    Parameters:
    - moviefile: Path to the input video file.
    - x0s: List of starting x-coordinates for cropping.
    - x1s: List of ending x-coordinates for cropping.
    - y0s: List of starting y-coordinates for cropping.
    - y1s: List of ending y-coordinates for cropping.
    - t0: Starting frame number for cropping.
    - t1: Ending frame number for cropping.
    - fps: Frames per second for the output video.
    """

    moviefile = kwargs['moviefile']
    x0s = kwargs['x0s']
    x1s = kwargs['x1s']
    y0s = kwargs['y0s']
    y1s = kwargs['y1s']
    t0 = kwargs['t0']
    t1 = kwargs['t1']
    x0_recenter = kwargs['x0_recenter']
    force = kwargs.get('force', False)

    success = False
    
    # open video file
    cap = cv2.VideoCapture(moviefile)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return success

    # get video properties 
    videoinfo = get_video_info(moviefile,cap)       
    fps = videoinfo['fps']
    
    for i in range(len(x0s)):
        for j in range(len(y0s)):
            c0 = x0s[i]
            c1 = x1s[i]
            r0 = y0s[j]
            r1 = y1s[j]
            outfile = get_cropvideo_name(moviefile,c0,c1,r0,r1)
            print(outfile)
            if os.path.isfile(outfile) and not force:
                print(f"File {outfile} already exists. Not overwriting.")
                continue

            # open video for writing with OpenCV, mjpeg codec
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(outfile, fourcc, fps, (c1-c0, r1-r0))
            if not out.isOpened():
                print("Error: Could not open video for writing.")
                sys.exit()
            # seek to frame t0
            cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
            # read frames and write to output file
            for frame_number in tqdm.trange(t0, t1, unit='frame', leave=False):
                ret, frame = cap.read()
                if ret:
                    # crop and recenter frame
                    frame_rgb = frame
                    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = recenter_frame(frame_rgb,x0_recenter)
                    cropped_frame = frame_rgb[r0:r1, c0:c1]
                    out.write(cropped_frame)
                else:
                    print(f"Error: Could not read frame {frame_number}")
                    break
            # release video writer
            out.release()

            # # Region to extract (width:height:x:y)
            # crop = f"{c1-c0}:{r1-r0}:{c0}:{r0}" 

            # # FFmpeg command
            # command = [
            #     'ffmpeg',
            #     '-i', moviefile,
            #     '-filter:v', f'crop={crop}',  # Crop filter
            #     '-c:v', 'mjpeg',  # MJPEG codec
            #     '-q:v', '2',  # Quality (2 is high quality, lower is better)
            #     '-an',  # No audio
            #     '-y',  # Overwrite output file
            #     outfile
            # ]
            # print(' '.join(command))
            # result = subprocess.run(command, check=True, stderr=subprocess.PIPE, universal_newlines=True)

    cap.release()

def moviefile_to_id(moviefile):
    """
    Convert a movie file name to a unique ID.
    
    Parameters:
    - moviefile: Path to the input video file.
    
    Returns:
    - id: Unique ID for the movie file.
    """
    moviename = os.path.basename(moviefile)
    m = re.match(r'^(.*_\d{3})[a-zA-Z\.][^_]*', moviename)
    if m is None:
        return None
    else:
        return m.group(1)


def parse_cropmoviefile(cropmoviefile):
    moviename = os.path.basename(cropmoviefile)
    m = re.match(r'^(?P<movieid>.*_[0-9]{3}).*_r(?P<r0>[0-9]+)(to|:)(?P<r1>[0-9]+)_c(?P<c0>[0-9]+)(to|:)(?P<c1>[0-9]+)\.avi$', moviename)
    if m is None:
        return None
    else:
        return {
            'movieid': m.group('movieid'),
            'r0': int(m.group('r0')),
            'r1': int(m.group('r1')),
            'c0': int(m.group('c0')),
            'c1': int(m.group('c1')),
        }

def read_crop_info(jsonfile):
    """
    Read crop info from a JSON file.
    
    Parameters:
    - jsonfile: Path to the JSON file containing crop info.
    
    Returns:
    - cropinfo: List of dictionaries containing crop info.
    """
    with open(jsonfile, 'r') as f:
        cropinfo = json.load(f)
    cropinfo = futurify_crop_info(cropinfo)
    
    return cropinfo    
    
def futurify_crop_info(cs):
    for c in cs:
        if 'y0_crop' in c:
            c['y0s'] = [c['y0_crop'],]
            c['y1s'] = [c['y1_crop'],]
            _ = c.pop('y0_crop')
            _ = c.pop('y1_crop')
    return cs
    
def get_crop_info(moviefile,jsonfile):
    """
    Get crop info for a given movie file from a JSON file.
    
    Parameters:
    - moviefile: Path to the input video file.
    - jsonfile: Path to the JSON file containing crop info.
    
    Returns:
    - cropinfo: Dictionary containing crop info for the specified movie file.
    """
    
    cropinfo = read_crop_info(jsonfile)
    
    for c in cropinfo:
        if c['moviefile'] == moviefile:
            return c
    return None

def print_crop_info(c):
    print(c['moviefile'])
    print(f"  x0s: {c['x0s']}")
    print(f"  x1s: {c['x1s']}")
    print(f"  y0s: {c['y0s']}")
    print(f"  y1s: {c['y1s']}")
    print(f"  t0: {c['t0']}")
    print(f"  t1: {c['t1']}")
    print(f"  x0_recenter: {c['x0_recenter']}")

def print_all_crop_info(jsonfile=None,cropinfo=None):
    """
    Print all crop info from a JSON file.
    
    Parameters:
    - jsonfile: Path to the JSON file containing crop info. Default = None
    - cropinfo: List of dictionaries containing crop info. If None, it will be read from the JSON file. Default = None.
    """
    if cropinfo is None:
        cropinfo = read_crop_info(jsonfile)
    
    for c in cropinfo:
        print_crop_info(c)
        
def save_crop_info(cropinfo,jsonfile):
    """
    Save crop info to a JSON file.
    
    Parameters:
    - cropinfo: List of dictionaries containing crop info.
    - jsonfile: Path to the JSON file to save the crop info.
    """
    with open(jsonfile, 'w') as f:
        json.dump(cropinfo, f, indent=4)

def main(moviefile,jsonfile, force=False):
    
    cropinfo = get_crop_info(moviefile, jsonfile)
    if cropinfo is None:
        print(f'No crop info found for {moviefile}, enter below')
        return
    
    # Call the crop_video function with the extracted parameters
    crop_video(**cropinfo,force=force)

if __name__ == "__main__":
    # Check if the script is run with the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python process_videos.py <moviefile> <jsonfile>")
        sys.exit(1)
    
    moviefile = sys.argv[1]
    jsonfile = sys.argv[2]

    # Check if the input video file exists
    if not os.path.isfile(moviefile):
        print(f"Error: The file {moviefile} does not exist.")
        sys.exit(1)
    # Check if the JSON file exists
    if not os.path.isfile(jsonfile):
        print(f"Error: The file {jsonfile} does not exist.")
        sys.exit(1)
    # Call the main function with the provided arguments
    main(moviefile, jsonfile)

