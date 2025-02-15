import glob
import hashlib
import os

import jump
import threading
import time

dictHash={}
gLock = threading.RLock()

def _buildNewSource(source,file,numSourceProc,updateCmdPipeIn_list,ioFlagList):
    global dictHash
    hashKey = int(hashlib.md5(source.encode('UTF-8')).hexdigest(), 16)
    gLock.acquire()
    while hashKey in dictHash:
        hashKey += 1
    idSource = jump.hash(hashKey, numSourceProc)
    dictHash[hashKey] = file
    gLock.release()
    updateCmdPipeIn_list[idSource].put((0, (hashKey, file)))
    ioFlag=ioFlagList[idSource]
    with ioFlag.get_lock():
        ioFlag.value += 1

def _dynamicLoading(numSourceProc,updateCmdPipeIn_list,files,ioFlagList):
    flagContinue=True
    while flagContinue:
        gLock.acquire()
        numValid = numSourceProc - len(dictHash)
        gLock.release()
        if numValid <= 0:
            time.sleep(5)
        else:
            for index in range(numValid):
                if len(files)>0:
                    print('files:',files,numValid)
                    # video_files_dir = files[0]
                    # for root_dir, child_dir, child_files in os.walk(video_files_dir):
                    #     for video_name in child_files:
                    #         file = os.path.join(root_dir,video_name)
                    #         if file.endswith('.mp4'):
                    #             move_files = [file]
                    file = files[0]
                    source = 'local:' + file
                    _buildNewSource(source, file, numSourceProc, updateCmdPipeIn_list, ioFlagList)
                    files.pop(0)
                else:
                    print('\n所有待检任务均已推送\n')
                    flagContinue = False
                    break
            
def taskManage(config,updateCmdPipeIn_list,queueResults,taskManagedata_queue,ioFlagList):
    numSourceProc = config.numSourceProc
    global dictHash

    files = []
    # files = glob.glob(pathname=config.sources)
    allexit = []
    for exit_root, exit_dirs, exit_files in os.walk(config.outputDir):
        for exi_file in exit_files:
            allexit.append(exi_file)
    
    for root, dirs, video_files in os.walk(config.sources):
        for file in video_files:
            file_path = os.path.join(root,file)
            npy_file = file.replace('.mp4', '_series.npy')
            print('===========',file.replace('.mp4', '_series.npy'))
            if file in allexit and npy_file in allexit:
                continue
            files.append(file_path)
    # print(files)
    '''
    for file in files:
        source = 'local:' + file
        _buildNewSource(source, file, numSourceProc, updateCmdPipeIn,ioFlagList)
    '''
    thread_dynamicLoading = threading.Thread(target=_dynamicLoading, args=(numSourceProc,\
                                                                           updateCmdPipeIn_list,\
                                                                           files, ioFlagList))
    thread_dynamicLoading.start()

    while True:
        protcl,contents = queueResults.get()
        if protcl == 0:
            for item in contents:
                source=dictHash[item]
                print(source)
                print(contents[item])
        elif protcl == 1:
            if len(contents) > 0:
                for item in contents:
                    if item in dictHash:
                        print(f'Job-`{dictHash[item]}` finished.')
                        gLock.acquire()
                        del dictHash[item]
                        gLock.release()


        '''  待解决：存在各进程不同步的情况，此时有可能最末时间段在其它进程首先检完，则如下删除语句会将正在检的key值删掉。
        elif protcl == 1:
            if len(contents) > 0:
                pipeL2Cache.send((1, contents))
                gLock.acquire()
                for item in contents:
                    if item in dictHash:
                        print(f'Job-`{dictHash[item]}` finished.')
                        del dictHash[item]
                gLock.release()

                print('\ntask Existed:')
                print(dictHash)
        '''
    return




    '''
    numFilesLastBatch = len(files) % config.batchSize
    perProcNumLastBatch = int(numFilesLastBatch / config.sourceProcNum)
    # 使用list存储最后一批中，每个文件处理进程管理的文件数：
    numFilesProcLast = [perProcNumLastBatch for _ in range(config.sourceProcNum)]
    for i in range(numFilesLastBatch % config.sourceProcNum):
        numFilesProcLast[i] += 1
    numBatch = int(len(files) / config.batchSize)
    perProcNumPerBatch = int(config.batchSize / config.sourceProcNum)
    # 使用list存储非最后一批的每批中，每个文件处理进程管理的文件数：
    numFilesProcPerBatch = [perProcNumPerBatch for _ in range(config.sourceProcNum)]
    for i in range(config.batchSize % config.sourceProcNum):
        numFilesProcPerBatch[i] += 1
    # 使用list存储每个文件处理进程管理的总文件数：
    numFilesProc = []
    for i in range(config.sourceProcNum):
        numFilesProc.append(numFilesProcLast[i] + numBatch * numFilesProcPerBatch[i])
    return numFilesLastBatch,numFilesProcLast,numBatch,numFilesProcPerBatch,numFilesProc
    '''