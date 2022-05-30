import matplotlib.pyplot as plt
import cantrips as can
import numpy as np

if __name__=="__main__":
    throughputs_dir = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/skySpectrograph/calibrationDataFiles/'
    throughput_from_mono = 'OSELOTS_throughput_2022_04_21.txt'
    throughput_from_laser = 'OSELOTS_throughput_2022_05_04.txt'
    save_fig_name = 'OSELOTS_throughput_from_mono_vs_from_laser.pdf'
    SToN_for_good_point = 2

    mono_throughput = can.readInColumnsToList(throughputs_dir + throughput_from_mono, delimiter = ',', n_ignore = 1, convert_to_float = 1)
    good_mono_throughputs = np.where(np.array(mono_throughput[1]) / np.array(mono_throughput[2]) > SToN_for_good_point)[0]
    print ('good_mono_throughputs = ' + str(good_mono_throughputs))
    mono_throughput = [np.array([col[i] for i in good_mono_throughputs]) for col in mono_throughput]
    laser_throughput = can.readInColumnsToList(throughputs_dir + throughput_from_laser, delimiter = ',', n_ignore = 1, convert_to_float = 1)
    good_laser_throughputs = np.where(np.array(laser_throughput[1]) / np.array(laser_throughput[2]) > SToN_for_good_point)[0]
    laser_throughput = [np.array([col[i] for i in good_laser_throughputs]) for col in laser_throughput]
    print ('good_laser_throughputs = ' + str(good_laser_throughputs))

    mono_plot = plt.scatter(mono_throughput[0], 1.0 / mono_throughput[1], c = 'k')
    plt.errorbar(mono_throughput[0], 1.0 / mono_throughput[1], yerr = mono_throughput[2] / (mono_throughput[1] ** 2.0), ecolor = 'k', fmt = 'none')
    laser_plot = plt.scatter(laser_throughput[0], 1.0 / laser_throughput[1], c = 'r')
    plt.errorbar(laser_throughput[0], 1.0 / laser_throughput[1], yerr = laser_throughput[2] / (laser_throughput[1] ** 2.0), ecolor = 'r', fmt = 'none')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('OSELOTS Throughput (ADU/s/Ry)')
    plt.ylim()
    plt.legend([mono_plot, laser_plot], ['Throughput from monochromator', 'Throughput from laser diodes'])
    plt.title('OSELOTS Throughput (2 ways)') 
    plt.savefig(throughputs_dir + save_fig_name)
    plt.show()
