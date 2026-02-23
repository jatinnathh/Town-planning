import cv2
import math
import itertools
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Path to your best model
MODEL_PATH = r"C:\Users\Jatin\Desktop\Town Planning\runs\detect\runs\detect\town_planning_final_run4\weights\best.pt"

# CALIBRATION: Measure how many pixels = 1 cm on your camera feed at the fixed height
PIXELS_PER_CM = 20

# Confidence Threshold (Higher = fewer false detections)
CONF_THRESHOLD = 0.40
# =================================================

def main():
    print("⏳ Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded!")

    # Open Webcam (0 is usually the default laptop cam, try 1 if you have external)
    cap = cv2.VideoCapture(1)
    
    # Set Resolution (Optional: HD is better for reading text)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # 1. PREDICT
        # verbose=False keeps your console clean
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # 2. EXTRACT OBJECTS
        detected_objects = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            detected_objects.append({
                "name": cls_name,
                "center": (cx, cy),
                "box": (x1, y1, x2, y2)
            })

            # Draw Box (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw Label
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw Center Dot
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 3. CALCULATE & DRAW DISTANCES (All Pairs)
        # Using itertools to get every unique combination of 2 objects
        pairs = list(itertools.combinations(detected_objects, 2))

        for obj_A, obj_B in pairs:
            pt1 = obj_A["center"]
            pt2 = obj_B["center"]
            
            # Distance Math
            dist_px = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            dist_cm = dist_px / PIXELS_PER_CM
            
            # Optimization: Only draw lines if objects are somewhat close 
            # (Optional: prevents screen clutter if you have 20 objects)
            # if dist_cm > 50: continue 

            # Draw Line (Cyan)
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            # Midpoint for text
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            
            label = f"{dist_cm:.1f}"
            
            # Text Settings (Bold)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            
            # Text Background (White Box)
            (w, h), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(frame, (mid_x - 5, mid_y - h - 5), (mid_x + w + 5, mid_y + 5), (255, 255, 255), -1)
            
            # Text (Black)
            cv2.putText(frame, label, (mid_x, mid_y), font, scale, (0, 0, 0), thickness)

        # 4. SHOW FRAME
        cv2.imshow("Town Planning AR System", frame)

        # quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 System closed.")

if __name__ == "__main__":
    main()