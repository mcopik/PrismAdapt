//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package prism;

public final class Preconditions
{
	static public <T> void checkNotNull(T object)
	{
		checkNotNull(object, "");
	}

	static public <T> void checkNotNull(T object, String msg)
	{
		if (object == null) {
			throw new NullPointerException(msg);
		}
	}

	static public void checkCondition(boolean condition)
	{
		checkCondition(condition, "");
	}

	static public void checkCondition(boolean condition, String msg)
	{
		if (!condition) {
			throw new IllegalArgumentException(msg);
		}
	}

	static public void checkIndex(int index, int size)
	{
		checkIndex(index, size, "");
	}

	static public void checkIndex(int index, int size, String msg)
	{
		if (index < 0 || index >= size) {
			throw new IllegalArgumentException(msg);
		}
	}
};