curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "conversation": [
             {
               "role": "user",
               "content": [
                 {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                 {"type": "text", "text": "Describe this image."}
               ]
             }
           ]
         }'


curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{
           "conversation": [
             {
               "role": "user",
               "content": [
                 {"type": "video", "path": "Penguins_720p_3min.mp4"},
                 {"type": "text", "text": "Describe this video in detail"}
               ]
             }
           ]
         }'