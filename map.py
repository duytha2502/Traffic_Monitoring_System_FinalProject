# Upload video
uploaded_video = st.sidebar.file_uploader("Upload video", type= ["mp4", "mov", "avi", "asf", "m4v"])
tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete= False)

if st.sidebar.button("Start tracking", use_container_width=True):
    with st.spinner("Processing video, please wait..."):

        # Thực hiện phân tích video
        processed_video, vehicles_os, vehicle_count_each, vehicle_counts, vehicle_inside_area, congestion_rates, avg_speeds, occupancy_densities, congestion_log = processVideo(tfile.name, model, class_names, corA, corB, corC, corD, max_speed_num, scale, cgs_spd, cgs_ocp)

        # Lưu video đã xử lí
        saved_video, video_name = save_video(processed_video, uploaded_video.name, PROCESSED_FOLDER)

        # Lưu video vào MongoDB
        video_id = save_video_to_mongodb(saved_video, video_name)

        capture_speeding_object(video_name, vehicles_os, fsvo)

        # Lưu kết quả vào CSV
        save_data_to_mongodb( vehicle_count_each, vehicle_counts, vehicle_inside_area, avg_speeds, occupancy_densities, congestion_rates, video_name)

        # Lưu logs
        log_action("Upload", st.session_state["email"], f"User {st.session_state["email"]} uploaded video {video_id} to start tracking")
    
    st.subheader("Traffic Overview")
    # Hiển thị video đã xử lý
    with st.container(border=True, key='1'):
        col1db1, col2db1, col3db1, col4db1 = st.columns(4)
        with col1db1:
            plot_gauge(
                indicator_number=vehicle_counts[-1],
                indicator_color="blue",
                indicator_suffix="",
                indicator_title="Total Vehicles",
                max_bound=100,  
            )
        with col2db1:
            plot_gauge(
                indicator_number=round(sum(congestion_rates)/len(congestion_rates), 2),
                indicator_color="red",
                indicator_suffix="%",
                indicator_title="Congestion Rate",
                max_bound=100,  
            )
        with col3db1:
            plot_gauge(
                indicator_number=round(sum(avg_speeds)/len(avg_speeds), 2),
                indicator_color="green",
                indicator_suffix="km/h",
                indicator_title="Speed",
                max_bound=100,  
            )
        with col4db1:
            plot_gauge(
                indicator_number=round(sum(occupancy_densities)/len(occupancy_densities), 2),
                indicator_color="green",
                indicator_suffix="%",
                indicator_title="Occupancy",
                max_bound=100,  
            )
    with st.container(border=True, key='2'):
        col1db2, col2db2 = st.columns(2, gap='large')
        with col1db2:
            st.subheader("Processed Video")
            display_video_from_mongodb(video_id) 
        with col2db2:
            placeholder = st.empty()
        
        col1db3, col2db3 = st.columns(2, gap='large')
        with col1db3:
            create_map(lattitude, longtitude)   
        with col2db3:
            fig2 = plot_bottom_right(avg_speeds, occupancy_densities, congestion_rates)
            st.plotly_chart(fig2, use_container_width=True, key="data")

    with st.container(border=True, key='3'):    
        col1db4, col2db4 = st.columns([6,4], gap='large')
        with col1db4:

            fleet_data = [
                {"icon": "🚗", "count": vehicle_count_each["car"], "type": "Car"},
                {"icon": "🏍️", "count": vehicle_count_each["motorbike"], "type": "Motorcycle"},
                {"icon": "🚛", "count": vehicle_count_each["truck"], "type": "Truck"},
                {"icon": "🚌", "count": vehicle_count_each["bus"], "type": "Bus"},
            ]
            # Tạo layout card với Streamlit columns
            cols = st.columns(len(fleet_data))

            for col, data in zip(cols, fleet_data):
                with col:
                    # Tạo card
                    st.markdown(f"""
                    <div style="
                        color: #31333f; 
                        padding: 15px; 
                        border-radius: 8px; 
                        text-align: left; 
                        display: flex; 
                        align-items: center;
                        justify-content: center; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 42px; margin-right: 15px; text-color: #01364d">{data['icon']}</div>
                        <div>
                            <div style="font-size: 26px; font-weight: bold;">{data['count']}</div>
                            <div style="font-size: 16px; font-weight: bold;">{data['type']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            fig3 = plot_bottom_left2(vehicle_count_each)
            st.plotly_chart(fig3, use_container_width=True, key="vehicle_counts_each")
        with col2db4:
            log_df = pd.DataFrame(congestion_log)
            st.subheader("Congestion Log")
            st.dataframe(log_df, use_container_width=True)          
        
        plot_bottom_left(vehicle_inside_area, placeholder)
else:
    st.subheader("Traffic Overview")
    with st.container(border=True, key='4'):
        col1db1, col2db1, col3db1, col4db1 = st.columns(4)
        with col1db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="blue",
                indicator_suffix="",
                indicator_title="Total Vehicles",
                max_bound=100,  
            )
        with col2db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="red",
                indicator_suffix="%",
                indicator_title="Congestion Rate",
                max_bound=100,  
            )
        with col3db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="green",
                indicator_suffix="km/h",
                indicator_title="Speed",
                max_bound=100,  
            )
        with col4db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="green",
                indicator_suffix="%",
                indicator_title="Occupancy",
                max_bound=100,  
            )
    with st.container(border=True, key='2'):
        col1db2, col2db2 = st.columns(2, gap='large')
        with col1db2:
            st.subheader("Video")
            
            if uploaded_video is not None:
                tfile.write(uploaded_video.read())
                dem_vid = open(tfile.name, 'rb')
                demo_bytes = dem_vid.read()
                st.video(demo_bytes)
            else:
                st.info("No video uploaded yet. Please upload a video to preview it.")
            # display_video_from_mongodb(video_id) 
        with col2db2:
            placeholder = st.empty()
        
        col1db3, col2db3 = st.columns(2, gap='large')
        with col1db3:
            st.subheader("Camera Coordinate")
            st.info("Process video to show map")   
        with col2db3:
            fig2 = plot_bottom_right([0], [0], [0])
            st.plotly_chart(fig2, use_container_width=True, key="data")

    with st.container(border=True, key='3'):    
        col1db4, col2db4 = st.columns([6,4], gap='large')
        with col1db4:

            fleet_data = [
                {"icon": "🚗", "count": 0, "type": "Car"},
                {"icon": "🏍️", "count": 0, "type": "Motorcycle"},
                {"icon": "🚛", "count": 0, "type": "Truck"},
                {"icon": "🚌", "count": 0, "type": "Bus"},
            ]
            # Tạo layout card với Streamlit columns
            cols = st.columns(len(fleet_data))

            for col, data in zip(cols, fleet_data):
                with col:
                    # Tạo card
                    st.markdown(f"""
                    <div style="
                        color: #31333f; 
                        padding: 15px; 
                        border-radius: 8px; 
                        text-align: left; 
                        display: flex; 
                        align-items: center;
                        justify-content: center; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 42px; margin-right: 15px; text-color: #01364d">{data['icon']}</div>
                        <div>
                            <div style="font-size: 26px; font-weight: bold;">{data['count']}</div>
                            <div style="font-size: 16px; font-weight: bold;">{data['type']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            fig3 = plot_bottom_left2({'car':0})
            st.plotly_chart(fig3, use_container_width=True, key="vehicle_counts_each")
        with col2db4:
            log_df = pd.DataFrame([0])
            st.subheader("Congestion Log")
            st.dataframe(log_df, use_container_width=True)          
        
        plot_bottom_left([0], placeholder)