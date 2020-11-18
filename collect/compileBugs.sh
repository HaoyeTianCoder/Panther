dir=/Users/haoye.tian/Documents/University/project/defects4j_buggy
proj=Chart
for bugId in $(seq 1 26)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done

proj=Closure
for bugId in $(seq 1 133)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done

proj=Lang
for bugId in $(seq 1 65)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done

proj=Math
for bugId in $(seq 1 106)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done

proj=Mockito
for bugId in $(seq 1 33)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done

proj=Time
for bugId in $(seq 1 27)
do
	cd ${proj}_${bugId}
	# defects4j compile
	defects4j test
	echo ${proj}_${bugId}
	cd ..
done
