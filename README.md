# yolov5.0_trt_self_official_api
yolov5-5.0+TensorRT-7.2.2.3+python+C++

环境配置：
	
	所有的安装包在百度盘，大家自取：

		链接: https://pan.baidu.com/s/1rTE__7NIBcS85M3c0QW-XA  密码: 083d
		--来自百度网盘超级会员V4的分享


	## 显卡驱动nvidia-driver-465、cuda以及cudnn的下载安装

	显卡驱动nvidia-driver-465、cuda以及cudnn的下载安装可以看我的这篇博客：


		https://blog.csdn.net/weixin_43269994/article/details/109030404
	## TensorRT-7.2.2.3下载安装

	在官网下载TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz

		tar xzvf TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz
	解压后先配置环境变量：

		sudo vim ~/.bashrc
	进入后，在最底部添加环境变量：

		export TRT_PATH=/home/lindsay/TensorRT-7.2.2.3
		export PATH=$PATH:$TRT_PATH/bin
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/lib
		export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_PATH/targets/x86_64-linux-gnu/lib

	进入到解压路径下的python文件夹，使用pip安装：


		cd TensorRT-7.2.2.3/python
		pip3 install tensorrt-7.2.2.3-cp37-none-linux_x86_64.whl

	安装uff以支持tensorflow


		cd TensorRT-7.2.2.3/uff
		pip3 install uff-0.6.9-py2.py3-none-any.whl

	安装graphsurgeon以支持自定义结构

		cd TensorRT-7.2.2.3/graphsurgeon
		pip3 install graphsurgeon-0.4.5-py2.py3-none-any.whl

	安装onnx_graphsurgeon以支持onnx

		cd TensorRT-7.2.2.3/onnx_grahsurgeon
		pip3 install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl






	## OpenCV下载安装

	下载地址OpenCV官网，选择最新的4.4.0版本(如果下载速度太慢，复制链接地址，使用迅雷)

		https://opencv.org/releases/


	编译与安装

	安装cmake
	OpenCV需要使用cmake进行编译

			sudo apt-get install cmake

	安装依赖

		sudo apt-get install build-essential pkg-config libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev

	出现以下问题：

		The following packages have unmet dependencies:
		 libavcodec-dev : Depends: libavutil-dev (= 7:3.4.2-2) but it is not going to be installed
			  Depends: libswresample-dev (= 7:3.4.2-2) but it is not going to be installed
		 libavformat-dev : Depends: libavformat57 (= 7:3.4.2-2) but it is not going to be installed
			   Depends: libavutil-dev (= 7:3.4.2-2) but it is not going to be installed
			   Depends: libswresample-dev (= 7:3.4.2-2) but it is not going to be installed
		 libgtk2.0-dev : Depends: libglib2.0-dev (>= 2.27.3) but it is not going to be installed
			 Depends: libgdk-pixbuf2.0-dev (>= 2.21.0) but it is not going to be installed
			 Depends: libpango1.0-dev (>= 1.20) but it is not going to be installed
			 Depends: libatk1.0-dev (>= 1.29.2) but it is not going to be installed
			 Depends: libcairo2-dev (>= 1.6.4-6.1) but it is not going to be installed
			 Depends: libx11-dev (>= 2:1.0.0-6) but it is not going to be installed
			 Depends: libxext-dev (>= 1:1.0.1-2) but it is not going to be installed
			 Depends: libxinerama-dev (>= 1:1.0.1-4.1) but it is not going to be installed
			 Depends: libxi-dev (>= 1:1.0.1-4) but it is not going to be installed
			 Depends: libxrandr-dev (>= 2:1.2.99) but it is not going to be installed
			 Depends: libxcursor-dev but it is not going to be installed
			 Depends: libxfixes-dev (>= 1:3.0.0-3) but it is not going to be installed
			 Depends: libxcomposite-dev (>= 1:0.2.0-3) but it is not going to be installed
			 Depends: libxdamage-dev (>= 1:1.0.1-3) but it is not going to be installed
			 Recommends: python (>= 2.4) but it is not going to be installed
		 libjpeg-dev : Depends: libjpeg8-dev but it is not going to be installed
		libswscale-dev : Depends: libavutil-dev (= 7:3.4.2-2) but it is not going to be installed
			  Depends: libswscale4 (= 7:3.4.2-2) but 7:3.4.8-0ubuntu0.2 is to be installed
		 libtiff5-dev : Depends: libtiff5 (= 4.0.9-5) but 4.0.9-5ubuntu0.4 is to be installed
		E: Unable to correct problems, you have held broken packages.

	使用aptitude

	aptitude与 apt-get 一样，是 Debian 及其衍生系统中功能极其强大的包管理工具。与 apt-get 不同的是，aptitude在处理依赖问题上更佳一些。举例来说，aptitude在删除一个包时，会同时删除本身所依赖的包。这样，系统中不会残留无用的包，整个系统更为干净。


		sudo aptitude install install build-essential pkg-config libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libswscale-dev libtiff5-dev

	运行后，不接受未安装方案，接受降级方案。


	解压

		unzip opencv-4.4.0

	 进入文件目录，创建build目录并进入

		cd opencv-4.4.0/
		mkdir build
		cd build

	使用cmake生成makefile文件

		cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_GTK=ON -D OPENCV_GENERATE_PKGCONFIG=YES ..

	CMAKE_BUILD_TYPE=RELEASE：表示编译发布版本
	CMAKE_INSTALL_PREFIX：表示生成动态库的安装路径，可以自定义
	WITH_GTK=ON：这个配置是为了防止GTK配置失败：即安装了libgtk2.0-dev依赖，还是报错未安装
	OPENCV_GENERATE_PKGCONFIG=YES：表示自动生成OpenCV的pkgconfig文件，否则需要自己手动生成。
	编译

		make -j8

	-j8表示使用多个系统内核进行编译，从而提高编译速度，不清楚自己系统内核数的，可以使用make -j$(nproc)
	如果编译时报错，可以尝试不使用多个内核编译，虽然需要更长的编译时间，但是可以避免一些奇怪的报错
	安装

		sudo make install

	注：如果需要重新cmake，请先将build目录下的文件清空，再重新cmake，以免发生错误
	环境配置

	## 将OpenCV的库添加到系统路径

	方法一：配置ld.so.conf文件

		sudo vim /etc/ld.so.conf

	在文件中加上一行 include /usr/local/lib，这个路径是cmake编译时填的动态库安装路径加上/lib
	配置ld.so.conf文件

	方法二：手动生成opencv.conf文件

		sudo vim /etc/ld.so.conf.d/opencv.conf

	是一个新建的空文件，直接添加路径，同理这个路径是cmake编译时填的动态库安装路径加上/lib

		/usr/local/lib

	以上两种方法配置好后，执行如下命令使得配置的路径生效

		sudo ldconfig

	配置系统bash
	因为在cmake时，选择了自动生成OpenCV的pkgconfig文件，在/usr/local/lib/pkgconfig路径可以看到文件
	opencv4.pc

	确保文件存在，执行如下命令

		sudo vim /etc/bash.bashrc

	在文末添加

		PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
		export PKG_CONFIG_PATH

	如下：

		bash.bashrc

	保存退出，然后执行如下命令使配置生效

		source /etc/bash.bashrc

	至此，Linux\Ubuntu18.04环境下OpenCV的安装以及配置已经全部完成，可以使用以下命令查看是否安装和配置成功


		pkg-config --modversion opencv4
		pkg-config --cflags opencv4
		pkg-config --libs opencv4



	## 使用TensorRT

	下载yolov5和tensorrtx


		git clone https://github.com/wang-xinyu/tensorrtx.git
		git clone https://github.com/ultralytics/yolov5.git


	将tensorrt/yolov5拷贝至yolov5下：sudo cp -r tenorrtx/yolov5 yolov5
	生成pt对应的wts: python gen_wts.py
	将wts放至build同级目录

	修改CMakeLists.txt：

		# tensorrt

		#include_directories(/usr/include/x86_64-linux-gnu/)
		#link_directories(/usr/lib/x86_64-linux-gnu/)

		include_directories(/home/lindsay/TensorRT-7.2.2.3/include)
		link_directories(/home/lindsay/TensorRT-7.2.2.3/lib)

	修改yololayer.h：

		static constexpr int CLASS_NUM = 80;#根据自己的类别修改

	开始编译并测试：

		mkdir build && cd build
		cmake ..
		make -j6
		sudo ./yolov5 -s ../yolov5s.wts ../yolov5s.engine s# 生成引擎
		sudo ./yolov5 -d ../yolov5s.engine ../samples#测试c++
		python3 yolov5_trt.py  #测试python

现在介绍一下本项目文件结构：

       mp4文件夹：存储6个mp4格式的视频
       samples文件夹：存储若干个jpg图片
       weights文件夹：存放yolov5-5.0的官方pt模型
       
将yolov5-5.0的官方dataset添加进了推理器：
在infer_api.py中，设置了一个flag：

	flag = 'official_images'  #   'image'  单张图片推理,     'official_images'   添加了官方的加载接口

 'image'是源码自带的推理接口，'official_images' 是我加的官方的dataset。
 我将官方是接口封装了一下，命名为：infer_load_offical_images_videos_single()
 我写了一个函数（judge_rtsp）作为判断输入源是RTSP还是本地视频的依据，然后根据判断结果选择dataset加载方式：

	source_flag = judge_rtsp(source, 'rtsp://admin')
    if source_flag:
        print('loading RTSP ...')
        dataset = LoadStreams(source, img_size=640)
    else:
        print('loading Videos ...')
        dataset = LoadImages(source, img_size=640)
 项目QQ群：561945362
 微信群：加微信LS932695342，直接拉进群
