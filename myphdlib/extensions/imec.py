def createProbeConfig(dst, reference='tip'):
    """
    """
    if reference == 'external':
        referenceNumber = 0
    elif reference == 'tip':
        referenceNumber = 1
    else:
        referenceNumber = 0

    #
    data = ['(0,384)']
    nActiveChannels = 0
    iLoop = 0
    while True:

        #
        if nActiveChannels >= 384:
            break

        #
        channelIndices = [
            iLoop * 4,
            iLoop * 4 + 3
        ]

        #
        bankNumbers = (
            0 if channelIndices[0] <= 383 else 1,
            0 if channelIndices[1] <= 383 else 1
        )

        #
        for i, channelIndex in enumerate(channelIndices):
            if channelIndex > 383:
                channelIndices[i] = channelIndex - 384

        #
        for channelIndex, bankNumber in zip(channelIndices, bankNumbers):
            entry = f'({channelIndex} {bankNumber} {referenceNumber} 500 250 1)'
            data.append(entry)
        nActiveChannels += 2
        iLoop += 1
    
    #
    data.append('\n')
    line = ''.join(data)
    with open(dst, 'w') as stream:
        stream.write(line)
        

    return