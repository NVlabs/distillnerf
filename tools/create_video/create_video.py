
def create_video(image_lst=[], video_file="output_video.avi"):
    import cv2
    frame_rate = 3  # Adjust as needed
    # import pdb; pdb.set_trace()

    frame_height, frame_width, _ = cv2.imread(image_lst[0]).shape
    # frame_width = 228 * 3  # Adjust based on your image width
    # frame_height = 128 * 4  # Adjust based on your image height

    # Initialize video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (800, 900))
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

    # Loop through each image sequence and add it to the video
    for i in range(0, len(image_lst)):
        img_path = image_lst[i]
        frame = cv2.imread(img_path)
        # import pdb; pdb.set_trace()
        video_writer.write(frame)

    # Release the video writer
    cv2.destroyAllWindows()
    video_writer.release()

# fig_num = 5
init_frame = 0
end_frame = 33
# direction = 'y' # forward
# direction = 'x' # rightward
# direction = 'z' # upward
# direction = 'rotate' # upward



image_lst = [f"./combine/{i}.png"for i in range(init_frame, end_frame)]
create_video(image_lst, video_file=f"./ours.mov")

# ffmpeg -i ours.mov -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer" -loop 0 ours.gif
#