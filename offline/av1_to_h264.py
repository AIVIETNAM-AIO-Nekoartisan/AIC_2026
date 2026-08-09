import ffmpeg
import sys

def convert_av1_to_h264(input_path: str, output_path: str):
    """
    Chuyển đổi một video từ codec AV1 sang H.264.

    Args:
        input_path (str): Đường dẫn đến file video AV1 đầu vào.
        output_path (str): Đường dẫn để lưu file video H.264 đầu ra.
    """
    print(f"Bắt đầu chuyển đổi '{input_path}'...")

    try:
        # Tạo stream đầu vào
        input_stream = ffmpeg.input(input_path)

        # Tạo stream đầu ra với codec video là H.264 (libx264)
        # và sao chép audio gốc để không phải mã hóa lại, giúp tiết kiệm thời gian
        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            vcodec='libx264',  # Bộ mã hóa H.264
            acodec='copy'      # Sao chép luồng âm thanh, không mã hóa lại
        )

        # Thực thi lệnh ffmpeg
        # overwrite_output=True sẽ ghi đè file đầu ra nếu nó đã tồn tại
        ffmpeg.run(output_stream, overwrite_output=True, quiet=False)

        print(f"✅ Chuyển đổi thành công! File đã được lưu tại: '{output_path}'")
        return True

    except ffmpeg.Error as e:
        print("❌ Đã xảy ra lỗi trong quá trình chuyển đổi:", file=sys.stderr)
        # In ra thông tin lỗi từ ffmpeg để dễ dàng gỡ rối
        print(e.stderr.decode('utf8'), file=sys.stderr)
        return False

# --- SỬ DỤNG HÀM ---
if __name__ == "__main__":
    # Thay đổi tên file của bạn ở đây
    input_video = "animal_2.mp4"
    output_video = "animal_2_h264.mp4"

    convert_av1_to_h264(input_video, output_video)