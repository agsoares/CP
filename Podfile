platform :ios, '9.0'
use_frameworks!

target 'CP' do

  pod 'OpenCV', '~> 3.0.0'



end

post_install do |installer|
    installer.pods_project.build_configurations.each do |config|
        config.build_settings['GCC_OPTIMIZATION_LEVEL'] = '3'
    end
end